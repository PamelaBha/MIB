"""
Evaluation Module for interventions
"""
import sys
sys.path.append('/data/kebl6672/dpo-toxic-general/')

from typing import Dict

import os
import copy
import torch

# os.chdir('/code/dpo_toxic')

device = "cuda"

from toxicity.eval_interventions.eval_utils import (
    load_model,
    load_data,
    tokenize,
    get_intervene_name,
    pretty_print_results,
)
from toxicity.eval_interventions.generate_funcs import (
    generate_default,
    get_prompts,
    get_gold,
)
from toxicity.eval_interventions.metric_funcs import (
    run_f1,
    run_perplexity,
    run_perspective_api,
    run_dummy,
    run_detoxify_toxicity,
)
from toxicity.eval_interventions.hook_utils import (
    dont_hook,
    hook_subtract,
    scale_top_key_vectors,
    scale_top_value_vectors,
    scale_top_key_vectors_with_positive_activations,
    scale_top_value_vectors_with_positive_activations,
    # hook_and_revert_activations,
    # zero_out_activation_at_neuron,
    print_and_return_activation_values,
    assign_activations_to_neurons
)
from constants import (
    ROOT_DIR,
    PROFANITY,
    SEXUALLY_EXPLICIT,
    IDENTITY_ATTACK,
    THREAT,
    INSULT,
    SEVERE_TOXICITY,
    TOXICITY,
    PERSPECTIVE_API_ATTRIBUTES as ATTRIBUTES,
)
from utils import verbose_print, VERBOSE

DATA_DIR = os.path.join(ROOT_DIR, "data/intervene_data")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")


GENERATE_FUNCS = {
    "get_prompts": get_prompts,
    "get_gold": get_gold,
}
METRIC_FUNCS = {
    "f1": run_f1,
    "perplexity": run_perplexity,
    "dummy": run_dummy,
    "perspective_api": run_perspective_api,
    "detoxify": run_detoxify_toxicity,
}
HOOK_FUNCS = {
    "subtraction": hook_subtract,
    "scale_key_vector": scale_top_key_vectors,
    "scale_value_vector": scale_top_value_vectors,
    "scale_key_vector_with_positive_activation": scale_top_key_vectors_with_positive_activations,
    "scale_value_vector_with_positive_activation": scale_top_value_vectors_with_positive_activations,
    # "revert_activations": hook_and_revert_activations,
    # "zero_out_activation_at_neuron": zero_out_activation_at_neuron,
    "print_and_return_activation_values": print_and_return_activation_values,
    "assign_activations_to_neurons": assign_activations_to_neurons
}
UNHOOK_FUNCS = {}


def generate(model, data, intervene_config):
    """
    Test intervention on a specific metric.
    """
    return GENERATE_FUNCS.get(intervene_config["method"], generate_default)(
        model, data, intervene_config["params"]
    )


def run_metric(
    metric_type,
    model,
    data_obj,
    intervene_results: Dict[str, torch.LongTensor],
    config,
):
    """
    Calculate specific metric.

    :intervene_results: Mapping from intervention specification to a tensor
        of shape [data_size, max_prompt_len + max_new_tokens]
    """
    return METRIC_FUNCS[metric_type](
        model,
        data_obj,
        intervene_results,
        config,
    )


def hook_model(model, config):
    """
    Hook model.
    """
    hook = HOOK_FUNCS.get(config["method"], dont_hook)(model, config["params"])
    
    # Ensure the hook(s) are always returned as a list, and make sure they're not tuples
    if isinstance(hook, tuple):
        hook = list(hook)  # Convert tuple to a list if needed
    elif not isinstance(hook, list):
        hook = [hook]
    
    return model, hook  # Return both the model and the hook(s)




def unhook_model(model, hooks):
    """
    Remove hooks in the model. Ensure 'hooks' is iterable to avoid errors.
    """
    # Ensure 'hooks' is a list or tuple to safely iterate over it
    if not isinstance(hooks, (list, tuple)):
        hooks = [hooks]
    
    # Iterate over the hooks and remove each one
    for hook in hooks:
        if hasattr(hook, "remove"):
            hook.remove()  # Safely remove only if the hook has a 'remove' method
        else:
            print(f"Warning: Hook {hook} does not have a remove method.")



def _eval_intervene(
    model, tokenizer, model_config, intervene_config, metric_configs
):
    """
    Evaluation intervention on set of metrics.
    """
    assert "method" in intervene_config
    intervene_config["params"]["device"] = model_config["device"]

    results = {}
    for _metric_conf in metric_configs:
        metric_type = _metric_conf["metric"]
        intervene_config["params"]["max_new_tokens"] = None

        verbose_print(f"Evaluating {metric_type}")
        data = _metric_conf["tokenized"]

        intervene_config["params"]["hook_timesteps"] = -1
        if metric_type == "perplexity":
            intervene_config["params"]["hook_timesteps"] = 0

        _, hooks = hook_model(model, intervene_config)

        generations = {}
        do_generate = _metric_conf["generate"]
        if do_generate:

            intervene_config["params"]["max_new_tokens"] = _metric_conf[
                "max_new_tokens"
            ]
            intervene_config["params"]["batch_size"] = model_config[
                "batch_size"
            ]
            generations = generate(model, data, intervene_config)
            for gen in generations["pred_text"][:30]:
                verbose_print(gen)

        results[metric_type] = run_metric(
            metric_type,
            model,
            data,
            generations,
            _metric_conf.get("params"),
        )
        # unhook_model(model, hooks)
    return results


def unroll_intervene(configs):
    """
    Unroll any nested configurations.
    """
    unrolled = []
    for _config in configs:
        method = _config["method"]
        if method != "subtraction":
            unrolled.append(_config)
            continue

        params = _config["params"]
        scales = params.pop("scales", [])
        if len(scales) < 1:
            raise RuntimeError("Missing scale value?")

        subtract_sets = params.pop("subtract_from", [])
        if len(subtract_sets) < 1:
            raise RuntimeError("Missing subtract_from value?")

        for scale in scales:
            for subtract_set in subtract_sets:
                config_copy = copy.deepcopy(_config)
                config_copy["params"]["scale"] = scale
                config_copy["params"]["subtract_from"] = subtract_set
                unrolled.append(config_copy)

    return unrolled


def tokenize_data(tokenizer, config):
    """
    Tokenize all data beforehand.
    """
    metric_configs = config["metrics"]

    tokenized_data = {}
    for _metric_conf in metric_configs:
        datapath = _metric_conf["datapath"]
        if datapath in tokenized_data:
            _metric_conf["tokenized"] = tokenized_data[datapath]
            continue

        data = load_data(_metric_conf)
        tokenized_data[datapath] = tokenize(tokenizer, data, _metric_conf)
        _metric_conf["tokenized"] = tokenized_data[datapath]


def run_eval(config):
    """
    Run eval!
    """
    model_config = config["model"]
    metric_configs = config["metrics"]
    interventions = config["interventions"]

    assert len(metric_configs) == len(
        list(set([x["metric"] for x in metric_configs]))
    ), "Mismatch -- you likely specified the same metric twice!"

    model, tokenizer = load_model(model_config)
    model.tokenizer = tokenizer

    # Set padding side for LLaMA model (decoder-only models)
    tokenizer.padding_side = "left"

    # Ensure tokenizer has a padding token set before proceeding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize all data beforehand.
    for _metric_conf in metric_configs:
        if "params" not in _metric_conf:
            _metric_conf["params"] = {}
        _metric_conf["params"]["pad_token_id"] = tokenizer.pad_token_id
        _metric_conf["params"]["batch_size"] = model_config["batch_size"]
        _metric_conf["params"]["device"] = model_config["device"]

    tokenize_data(tokenizer, config)

    interventions = unroll_intervene(interventions)
    results = {}
    for intervene_config in interventions:

        intervene_name = get_intervene_name(intervene_config)
        verbose_print(f"  Evaluating intervention {intervene_name}")

        results[intervene_name] = _eval_intervene(
            model, tokenizer, model_config, intervene_config, metric_configs
        )
        pretty_print_results(results)
    return results


def main():
    """ Driver """
    verbose_mode = VERBOSE  
    config = {
        "model": {
            "model_or_path": "google/gemma-2-9b", #"meta-llama/Llama-3.1-8B", #"google/gemma-2-2b", # "gpt2-medium", # "meta-llama/Llama-3.1-8B", # "google/gemma-2-2b-it", # "mistralai/Mistral-7B-v0.1"
            # "state_dict_path": os.path.join(CKPT_DIR, "llama_dpo.pt"), # Use the DPO model # dpo.pt #mistral_dpo.pt
            "tokenizer": "google/gemma-2-9b", # "mistralai/Mistral-7B-v0.1", # "meta-llama/Llama-3.1-8B", # "meta-llama/Llama-2-7b-hf", #"google/gemma-2-2b-it", #"mistralai/Mistral-7B-v0.1",#"google/gemma-2-2b", #"meta-llama/Meta-Llama-3-8B", # gpt2-medium
            "batch_size": 64,
            "device": device if torch.cuda.is_available() else "cpu",
        },
        "metrics": [
            # {
            #     "datapath": os.path.join(DATA_DIR, "challenge_prompts_dev.jsonl") if verbose_mode else os.path.join(DATA_DIR, "challenge_prompts.jsonl"),
            #     "metric": "perspective_api",
            #     "max_prompt_size": 32,
            #     "max_new_tokens": 20,
            #     "generate": True,
            #     "params": {"attributes": ATTRIBUTES, "include_prompt": False},
            # },
            {
                "datapath": os.path.join(DATA_DIR, "challenge_prompts_dev.jsonl") if verbose_mode else os.path.join(DATA_DIR, "challenge_prompts.jsonl"),
                "metric": "detoxify",
                "max_prompt_size": 32,
                "max_new_tokens": 20,
                "generate": True,
                "params": {"include_prompt": False},  
            },
            {
                "datapath": "wikitext",
                "dataname": "wikitext-2-raw-v1",
                "split": "test",
                "metric": "perplexity",
                "generate": False,
            },
            {
                "datapath": os.path.join(DATA_DIR, "wiki_samples_dev.jsonl") if verbose_mode else os.path.join(DATA_DIR, "wiki_samples.jsonl"),
                "metric": "f1",
                "max_prompt_size": 32,
                "max_new_tokens": 20,
                "generate": True,
            },
        ],
        "interventions": [
            {"method": "noop", "params": {}},
            # {
            #     "method": "subtraction",
            #     "params": {
            #         "type": "mlp_w_out",
            #         "idx": 770,
            #         "layer": 19,
            #         "subtract_from": [[23]],
            #         "scales": [20],
            #     }
            # },
            # {
            #     "method": "subtraction",
            #     "params": {
            #         "type": "toxic_probe",
            #         "scales": [7],
            #         "subtract_from": [[31]], 
            #         "datapath": os.path.join(CKPT_DIR, "mistral_probe.pt"),
            #     }
            # },
            # {
            #     "method": "subtraction",
            #     "params": {
            #         "type": "svd",
            #         "idx": 0,
            #         "scales": [100],
            #         "subtract_from": [[23]],
            #         "topk_sorted_score": 512,
            #         "datapath": os.path.join(CKPT_DIR, "probe.pt"),
            #     }
            # },
            # {
            #      "method": "scale_key_vector", 
            #      "params": {
            #          "probe_vector_path": os.path.join(CKPT_DIR, "probe.pt"),
            #          "topk_sorted_score": 7,
            #          "scale_factor": 10
            #     }
            # },
            # {
            #      "method": "scale_value_vector", 
            #      "params": {
            #          "probe_vector_path": os.path.join(CKPT_DIR, "probe.pt"),
            #          "topk_sorted_score": 128,
            #          "scale_factor": 0
            #     }
            # }
            # {
            #      "method": "scale_key_vector_with_positive_activation", 
            #      "params": {
            #          "topk_sorted_score": 3000,
            #          "scale_factor": 0,
            #          "toxic_positive_acts_index_csv_path": "/code/dpo_toxic/toxic_positive_acts_idxs.csv"
            #     }
            # }
            # {
            #      "method": "scale_value_vector_with_positive_activation", 
            #      "params": {
            #          "topk_sorted_score": 36,
            #          "scale_factor": 0,
            #          "toxic_positive_acts_index_csv_path": "/data/kebl6672/dpo-toxic-neuron/toxic_positive_acts_idxs.csv"
            #     }
            # }
            # {
            #      "method": "revert_activations", 
            #      "params": {
            #         "probe_vector_path": os.path.join(CKPT_DIR, "probe.pt"),
            #          "topk_sorted_score": 1,
            #          "modification_value": [1]
            #          }
            # }
            # {
            #      "method": "zero_out_activation_at_neuron", 
            #      "params": {
            #         "layer_index": 19,
            #         "neuron_index": 770
            #          }
            # }
        #     {
        #          "method": "print_and_return_activation_values", 
        #          "params": {
        #             "layer_index": 19,
        #             "neuron_index": 770
        #              }
        #     }
        # # ],
            # {
            #      "method": "assign_activations_to_neurons", 
            #      "params": {
            #         "neuron_configs": [(19, 770, -0.0169772829954633), (12, 771, 0.0496953833745479), (18, 2669, 0.0038321316449479), (13, 668, -0.0705880833683915), (16, 255, -0.0011058073714611), (12, 882, -0.1133939705224594), (19, 1438, 0.1529453023995499), (9, 545, -0.1068636572948586), (8, 2854, -0.0539521720878145), (3, 3680, -0.0152621103135037), (14, 1958, -0.1336678191107452), (7, 1735, -0.1179427397625279), (13, 2258, -0.1093105478944804), (11, 1550, -0.1123083027197374), (3, 704, -0.1076519843238774), (10, 3477, -0.0938268208474648), (13, 1023, -0.0981822808883453), (13, 253, -0.1269961312546307), (10, 2936, -0.1509416461190814), (0, 2352, -0.0246489237264331), (7, 1916, -0.1375381919114841), (3, 3742, -0.0388831071625881), (11, 2844, -0.1999453823939173), (11, 4021, -0.0645628194619178), (11, 175, -0.0332215121624722), (19, 3341, -0.0400893745592313), (3, 1656, -0.1188541217771682), (5, 1744, -0.190154341923651), (7, 3358, -0.2532758612441755), (12, 1826, -0.1986192696881124), (16, 603, -0.0830782151805525), (11, 3414, -0.153440778357708), (11, 2617, -0.0823726907498043), (9, 340, -0.1548002486880796), (8, 3200, 0.0812913229606043), (16, 1741, -0.1079462936981306), (19, 2312, -0.0782193847853405), (13, 1544, 0.0027353818740762), (20, 3210, 0.045648079442006), (12, 3413, -0.0975113355043999), (12, 3349, -0.1557714752831307), (0, 3752, -0.1018487662246094), (11, 3437, -0.1596352596948458), (15, 1696, -0.1773008260673571), (2, 3998, -0.2404714201153616), (23, 1672, -0.1595538310134663), (12, 877, 0.0338804036851871), (7, 3701, -0.1229026476996437), (6, 2994, -0.1044292764511131), (6, 3972, 0.2822398754686984), (13, 4065, -0.0809127054085283), (0, 3393, 0.124103451606729), (9, 2758, -0.1218359721533706), (15, 4051, -0.0847701643804821), (7, 2018, -0.2226346393420013), (7, 2494, -0.1645104707818317), (3, 2765, -0.0628559626750846), (15, 511, -0.2261776700333639), (12, 2756, -0.0069512457073926), (1, 2057, -0.107255351936011), (20, 3123, -0.1257509274113616), (13, 3620, -0.12264608116911), (17, 3064, -0.1558872781333324), (6, 3821, -0.2540771689574123), (13, 3243, 0.0382593392343362), (16, 16, -0.11258952638377), (6, 113, -0.1098691111012015), (7, 166, -0.1229041706395413), (20, 551, -0.2000685081162675), (4, 2335, -0.1331873717734029), (9, 1028, -0.200842567785549), (10, 628, 0.0099433078457777), (8, 3310, -0.0561256909540801), (9, 2540, -0.1782357352300969), (0, 2723, -0.2391809851356392), (5, 4054, -0.094869781287538), (8, 3066, -0.1489452561674395), (15, 3116, -0.0262833454149286), (14, 414, -0.0911489605074575), (9, 2077, -0.1690406507709104), (2, 2935, -0.0987791257141655), (4, 144, -0.1137421934419787), (19, 505, -0.0937405571638789), (11, 2205, -0.1616713106533749), (7, 514, -0.1631108662158852), (13, 1916, -0.1674662379183105), (1, 1342, -0.0545511557240037), (4, 3494, -0.1717189616623148), (11, 2485, -0.220357967817863), (4, 3500, -0.2014962638759058), (14, 883, -0.1315343724364878), (9, 3567, -0.1834575744352863), (9, 371, -0.1913185602681325), (8, 273, -0.142385338455897), (10, 1635, -0.1817160951207739), (0, 2037, -0.1065813935090753), (10, 3674, -0.2069919705879918), (5, 23, -0.2637373480441918), (12, 619, -0.1427143797986305), (6, 2946, -0.1439116056239962), (6, 3121, -0.1082256536347625), (7, 3688, 0.026968601174122), (2, 610, -0.2473871394588991), (7, 2348, -0.1216439589879744), (5, 26, -0.1046245002671845), (23, 1390, -0.1779461403295692), (2, 2656, -0.027266331871823), (6, 2521, -0.1434271484112041), (9, 3523, -0.1271697659168651), (8, 2790, -0.181357008261903), (12, 3611, -0.1745784571249808), (0, 19, -0.1156685221910141), (17, 3162, -0.0146124889138183), (5, 1983, -0.1486558439086762), (12, 1859, -0.2206632696177201), (6, 1270, -0.12588566863507), (12, 1587, -0.1281805486184736), (11, 2776, -0.075983336629917), (13, 692, -0.1251795385714149), (11, 339, -0.1107276465253711), (12, 45, -0.1113319106025306), (9, 2278, -0.0918226563949548), (9, 138, -0.2050152499189), (10, 2899, -0.1051104551242834), (12, 3595, -0.0315233536704853), (14, 1856, -0.1771137760282355), (7, 3125, -0.1651225824922573), (1, 572, 0.0006069859673697)]
            #         }
            # }
        ],
    }
    results = run_eval(config)
    print("Final Results:")
    pretty_print_results(results)


if __name__ == "__main__":
    main()


