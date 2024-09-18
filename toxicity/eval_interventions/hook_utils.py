"""
Utility functions for hooking.
"""
from functools import partial
import torch
import torch.nn.functional as F
import pandas as pd
import ast


def get_svd_u_vec(model, toxic_vector, topk_sorted_score, U_idx):
    """
    Get the svd U vector
    toxic_vector: toxic_vector [d_model]
    topk_sorted_score: (int) vectors we want to get
    U_idx: which u vec
    """
    scores = []
    for layer in range(model.config.n_layer):
        # mlp_outs = model.blocks[layer].mlp.W_out
        # [d_mlp, d_model]
        mlp_outs = model.transformer.h[layer].mlp.c_proj.weight
        cos_sims = F.cosine_similarity(
            mlp_outs, toxic_vector.unsqueeze(0), dim=1
        )
        _topk = cos_sims.topk(k=100)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)

    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    top_vecs = [
        # model.blocks[x[2]].mlp.W_out[x[1]]
        model.transformer.h[x[2]].mlp.c_proj.weight[x[1]]
        for x in sorted_scores[:topk_sorted_score]
    ]
    top_vecs = [x / x.norm() for x in top_vecs]
    _top_vecs = torch.stack(top_vecs)

    svd = torch.linalg.svd(_top_vecs.transpose(0, 1))
    svd_U = svd.U.transpose(0, 1)
    return svd_U[U_idx]


def get_intervene_vector(model, config):
    """
    Get vector according to specifications in :config:
    """

    def _get_mlp_w_out(_config):
        layer = _config["layer"]
        idx = _config["idx"]
        return model.transformer.h[layer].mlp.c_proj.weight[idx]

    def _get_mlp_w_in(_config):
        w_in_idx = _config["w_ins"][0]
        layer = w_in_idx[0]
        idx = w_in_idx[1]
        return model.transformer.h[layer].mlp.c_fc.weight[:, idx]

    def _get_toxic_probe(_config):
        return torch.load(_config["datapath"])

    def _get_svd(_config):
        topk_sorted_score = _config["topk_sorted_score"]
        u_idx = _config["idx"]
        toxic_vector = torch.load(_config["datapath"])
        return get_svd_u_vec(model, toxic_vector, topk_sorted_score, u_idx)

    def _get_random(_config):
        shape = model.transformer.h[0].mlp.c_proj.weight[0].shape
        device = model.device
        return torch.rand(shape).to(device)

    return {
        "mlp_w_out": _get_mlp_w_out,
        "mlp_w_in": _get_mlp_w_in,
        "toxic_probe": _get_toxic_probe,
        "svd": _get_svd,
        "random": _get_random,
    }[config["type"]](config)


def hook_subtract(model, config):
    intervene_vector = get_intervene_vector(model, config)
    scale = config["scale"]
    subtract_from = config["subtract_from"]
    hook_timesteps = config["hook_timesteps"]

    def patch(vec, _scale):
        def hook(module, input, output):

            _vec = vec.unsqueeze(0).unsqueeze(0)
            if hook_timesteps == -1:
                _vec = _vec.repeat(output.shape[0], 1, 1)
            else:
                _vec = _vec.repeat(output.shape[0], output.shape[1], 1)
            output[:, hook_timesteps:, :] = output[:, hook_timesteps:, :] - (
                _scale * _vec
            )
            return output

        return hook

    hooks = []
    for layer in subtract_from:
        # hook = model.transformer.h[layer].mlp.c_proj.register_forward_hook(
        hook = model.transformer.h[layer].mlp.register_forward_hook(
            patch(intervene_vector, scale)
        )
        hooks.append(hook)
    return model, hooks


def dont_hook(model, config):
    return model, []


def scale_top_key_vectors(model, config):
    """
    Scales the key vectors of the MLPs based on the cosine similarity of the value vectors to a given toxic vector.
    Args:
        model (torch.nn.Module): The model being modified.
        config (dict): Configuration for scaling including the path to the toxic vector, top_k, and scale_factor.
    """
    toxic_vector = torch.load(config['probe_vector_path'])
    topk_sorted_score = config['topk_sorted_score']
    scale_factor = config['scale_factor']
    scores = []
    
    for layer in range(model.config.n_layer):
        value_vectors = model.transformer.h[layer].mlp.c_proj.weight
        cos_sims = F.cosine_similarity(value_vectors, toxic_vector.unsqueeze(0), dim=1)
        _topk = cos_sims.topk(k=1000)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)

    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    # print(sorted_scores[:20])

    top_key_vecs = [
        model.transformer.h[x[2]].mlp.c_fc.weight[:, x[1]]
        for x in sorted_scores[:topk_sorted_score]
    ]
    with torch.no_grad():
        for tensor in top_key_vecs:
            tensor *= scale_factor

    # Return model and an empty list of hooks for consistency
    return model, [] 



def scale_top_value_vectors(model, config):
    """
    Scales the value vectors of the MLPs based on the cosine similarity of the value vectors to a given toxic vector.
    Args:
        model (torch.nn.Module): The model being modified.
        config (dict): Configuration for scaling including the path to the toxic vector, top_k, and scale_factor.
    """
    toxic_vector = torch.load(config['probe_vector_path'])  # Load the toxic vector
    topk_sorted_score = config['topk_sorted_score']  # Number of top vectors to scale
    scale_factor = config['scale_factor']  # Factor by which to scale the vectors
    scores = []
    
    for layer in range(model.config.n_layer):
        # Now, target the value vectors in the MLP
        value_vectors = model.transformer.h[layer].mlp.c_proj.weight
        
        # Compute cosine similarities between the value vectors and the toxic vector
        cos_sims = F.cosine_similarity(value_vectors, toxic_vector.unsqueeze(0), dim=1)
        
        # Get the top k most similar value vectors
        _topk = cos_sims.topk(k=1000)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)

    # Sort the scores in descending order based on cosine similarity
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    
    # Select the top `topk_sorted_score` value vectors and scale them
    top_value_vecs = [
        model.transformer.h[x[2]].mlp.c_proj.weight[x[1], :]
        for x in sorted_scores[:topk_sorted_score]
    ]
    
    # Scale the selected value vectors
    with torch.no_grad():
        for tensor in top_value_vecs:
            tensor *= scale_factor

    return model, [] 




def scale_top_key_vectors_with_positive_activations(model, config):
    """
    Scales the key vectors for neurons with positive activations before DPO based on the ranks of their cosine similarity with the toxic probe.
    
    Args:
        model (torch.nn.Module): The model being modified.
        config (dict): Configuration for scaling including the path to the toxic vector, top_k, and scale_factor.
    """
    topk_sorted_score = config['topk_sorted_score']  
    scale_factor = config['scale_factor']  
    
    # Load the sorted scores from the CSV
    sorted_scores_df = pd.read_csv(config['toxic_positive_acts_index_csv_path'])
    
    # Select the top `topk_sorted_score` layer and neuron indices from the CSV
    top_layer_neuron_indices = sorted_scores_df.head(topk_sorted_score)
    
    # Scale the selected value vectors
    with torch.no_grad():
        for _, row in top_layer_neuron_indices.iterrows():
            layer_idx = row['layer_idx']
            neuron_idx = row['neuron_idx']
            tensor = model.transformer.h[layer_idx].mlp.c_fc.weight[:, neuron_idx]
            tensor *= scale_factor

    return model, [] 



def scale_top_value_vectors_with_positive_activations(model, config):
    """
    Scales the value vectors with positive activations before DPO based on the ranks of their cosine similarity with the toxic probe.
    
    Args:
        model (torch.nn.Module): The model being modified.
        config (dict): Configuration for scaling including the path to the toxic vector, top_k, and scale_factor.
    """
    topk_sorted_score = config['topk_sorted_score']  
    scale_factor = config['scale_factor']  
    
    # Load the sorted scores from the CSV
    sorted_scores_df = pd.read_csv(config['toxic_positive_acts_index_csv_path'])
    
    # Select the top `topk_sorted_score` layer and neuron indices from the CSV
    top_layer_neuron_indices = sorted_scores_df.head(topk_sorted_score)
    
    # Scale the selected value vectors
    with torch.no_grad():
        for _, row in top_layer_neuron_indices.iterrows():
            layer_idx = row['layer_idx']
            neuron_idx = row['neuron_idx']
            tensor = model.transformer.h[layer_idx].mlp.c_proj.weight[neuron_idx, :]
            tensor *= scale_factor

    return model, [] 




# def revert_activations(model, config):
#     """
#     Modify activations to pre-DPO states on most toxic value vectors.
#     """
#     topk_sorted_score = config['topk_sorted_score']  
    
#     # Load the sorted scores from the CSV
#     sorted_scores_df = pd.read_csv(config['toxic_neurons_with_key_vectors_and_bias_path'])
    
#     # Convert the 'key_vector' column from string to list
#     sorted_scores_df['key_vector'] = sorted_scores_df['key_vector'].apply(ast.literal_eval)
    
#     # Convert to tensors
#     sorted_scores_df['key_vector'] = sorted_scores_df['key_vector'].apply(torch.tensor)
    
#     # Select the top `topk_sorted_score` layer and neuron indices from the CSV
#     top_layer_neuron_indices = sorted_scores_df.head(topk_sorted_score)

#     # Modify the selected key vectors and bias
#     with torch.no_grad():
#         for _, row in top_layer_neuron_indices.iterrows():
#             layer_idx = row['layer_idx']
#             neuron_idx = row['neuron_idx']
#             old_key_vector = row['key_vector']
#             old_bias = torch.tensor(row['bias'])  # Ensure this is a tensor
#             key_vector = model.transformer.h[layer_idx].mlp.c_fc.weight[:, neuron_idx]
#             bias = model.transformer.h[layer_idx].mlp.c_fc.bias[neuron_idx]
            
#             # Modify key vector and bias
#             key_vector.copy_(old_key_vector)
#             bias.copy_(old_bias)

#     return model,[]





def hook_and_revert_activations(model, config):
    """
    Hook and modify activations based on cosine similarity with a toxic vector.
    """
    # Load the toxic vector
    print("Loading toxic vector...")
    toxic_vector = torch.load(config['probe_vector_path'])  # Load the toxic vector
    scores = []
    
    # Identify top-k similar value vectors across all layers
    print("Identifying top-k similar value vectors...")
    for layer in range(model.config.n_layer):
        print(f"Processing layer {layer}...")
        # Target the value vectors in the MLP
        value_vectors = model.transformer.h[layer].mlp.c_proj.weight
        
        # Compute cosine similarities between the value vectors and the toxic vector
        cos_sims = F.cosine_similarity(value_vectors, toxic_vector.unsqueeze(0), dim=1)
        
        # Get the top k most similar value vectors
        _topk = cos_sims.topk(k=config['topk_sorted_score'])
        _values = _topk.values.tolist()
        _idxs = _topk.indices.tolist()
        topk = list(zip(_values, _idxs, [layer] * len(_idxs)))
        scores.extend(topk)

    # Sort the scores in descending order based on cosine similarity
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    print(f"Top-k sorted scores identified: {sorted_scores[:5]}... (showing first 5)")

    # Hook function to modify activations
    def modify_activations(module, input, output):
        """
        A forward hook function to modify activations before they are passed to c_proj.
        """
        print("Modifying activations...")
        topk_sorted_score = config['topk_sorted_score']
        modification_value = config['modification_value']
        
        # Ensure modification_value is not longer than sorted_scores[:topk_sorted_score]
        if len(modification_value) > topk_sorted_score:
            modification_value = modification_value[:topk_sorted_score]

        # Get the input tensor
        activation_input = input[0]
        original_shape = activation_input.shape
        print(f"Original activation input shape: {original_shape}")

        # Modify the activations for the selected neurons
        with torch.no_grad():
            for i, score in enumerate(sorted_scores[:topk_sorted_score]):
                vector_idx = score[1]
                print(f"Modifying neuron {vector_idx} in activation...")

                # Ensure modification_value[i] is compatible with activation_input[:, :, vector_idx]
                if torch.is_tensor(modification_value[i]):
                    if modification_value[i].shape == ():
                        # If scalar, broadcast it
                        activation_input[:, :, vector_idx] = modification_value[i]
                    else:
                        # If tensor, ensure it's the correct shape
                        assert modification_value[i].shape == activation_input[:, :, vector_idx].shape, \
                            f"Shape mismatch: modification_value[{i}].shape = {modification_value[i].shape}, expected {activation_input[:, :, vector_idx].shape}"
                        activation_input[:, :, vector_idx] = modification_value[i]
                else:
                    # If modification_value[i] is a scalar, broadcast it to the appropriate shape
                    activation_input[:, :, vector_idx] = torch.full_like(activation_input[:, :, vector_idx], modification_value[i])

        modified_shape = activation_input.shape
        print(f"Modified activation input shape: {modified_shape}")

        # Ensure the modified shape matches the original shape
        assert modified_shape == original_shape, "Shape of activation input changed unexpectedly after modification!"

        print("Activation modification complete.")
        return activation_input  # Ensure the input is returned if modified

    # Register the forward hook for each layer in the model
    hooks = []
    for layer in range(model.config.n_layer):
        print(f"Registering hook for layer {layer}...")
        hook = model.transformer.h[layer].mlp.c_proj.register_forward_hook(modify_activations)
        hooks.append(hook)

    print("Hooks registered successfully.")
    return model, hooks  # Return both the model and the hooks