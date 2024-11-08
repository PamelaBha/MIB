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
            # print(f"Output shape: {output.shape}")
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





# def hook_and_revert_activations(model, config):
#     """
#     Hook and modify activations based on cosine similarity with a toxic vector.
#     """
#     # Load the toxic vector
#     print("Loading toxic vector...")
#     toxic_vector = torch.load(config['probe_vector_path'])  # Load the toxic vector
#     scores = []
    
#     # Identify top-k similar value vectors across all layers
#     print("Identifying top-k similar value vectors...")
#     for layer in range(model.config.n_layer):
#         print(f"Processing layer {layer}...")
#         # Target the value vectors in the MLP
#         value_vectors = model.transformer.h[layer].mlp.c_proj.weight
        
#         # Compute cosine similarities between the value vectors and the toxic vector
#         cos_sims = F.cosine_similarity(value_vectors, toxic_vector.unsqueeze(0), dim=1)
        
#         # Get the top k most similar value vectors
#         _topk = cos_sims.topk(k=config['topk_sorted_score'])
#         _values = _topk.values.tolist()
#         _idxs = _topk.indices.tolist()
#         topk = list(zip(_values, _idxs, [layer] * len(_idxs)))
#         scores.extend(topk)

#     # Sort the scores in descending order based on cosine similarity
#     sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
#     print(f"Top-k sorted scores identified: {sorted_scores[:5]}... (showing first 5)")

#     # Hook function to modify activations
#     def modify_activations(module, input, output):
#         """
#         A forward hook function to modify activations before they are passed to c_proj.
#         """
#         print("Modifying activations...")
#         topk_sorted_score = config['topk_sorted_score']
#         modification_value = config['modification_value']
        
#         # Ensure modification_value is not longer than sorted_scores[:topk_sorted_score]
#         if len(modification_value) > topk_sorted_score:
#             modification_value = modification_value[:topk_sorted_score]

#         # Get the input tensor
#         activation_input = input[0]
#         original_shape = activation_input.shape
#         print(f"Original activation input shape: {original_shape}")

#         # Modify the activations for the selected neurons
#         with torch.no_grad():
#             for i, score in enumerate(sorted_scores[:topk_sorted_score]):
#                 vector_idx = score[1]
#                 print(f"Modifying neuron {vector_idx} in activation...")

#                 # Ensure modification_value[i] is compatible with activation_input[:, :, vector_idx]
#                 if torch.is_tensor(modification_value[i]):
#                     if modification_value[i].shape == ():
#                         # If scalar, broadcast it
#                         activation_input[:, :, vector_idx] = modification_value[i]
#                     else:
#                         # If tensor, ensure it's the correct shape
#                         assert modification_value[i].shape == activation_input[:, :, vector_idx].shape, \
#                             f"Shape mismatch: modification_value[{i}].shape = {modification_value[i].shape}, expected {activation_input[:, :, vector_idx].shape}"
#                         activation_input[:, :, vector_idx] = modification_value[i]
#                 else:
#                     # If modification_value[i] is a scalar, broadcast it to the appropriate shape
#                     activation_input[:, :, vector_idx] = torch.full_like(activation_input[:, :, vector_idx], modification_value[i])

#         modified_shape = activation_input.shape
#         print(f"Modified activation input shape: {modified_shape}")

#         # Ensure the modified shape matches the original shape
#         assert modified_shape == original_shape, "Shape of activation input changed unexpectedly after modification!"

#         print("Activation modification complete.")
#         return activation_input  # Ensure the input is returned if modified

#     # Register the forward hook for each layer in the model
#     hooks = []
#     for layer in range(model.config.n_layer):
#         print(f"Registering hook for layer {layer}...")
#         hook = model.transformer.h[layer].mlp.c_proj.register_forward_hook(modify_activations)
#         hooks.append(hook)

#     print("Hooks registered successfully.")
#     return model, hooks  # Return both the model and the hooks








# def zero_out_activation_at_neuron(model, config):
#     """
#     Modify the activation coefficient for a specific neuron (neuron_idx) in the output of the c_fc layer (first weight vector for the MLP neuron).
#     """
#     layer_idx = config['layer_index']
#     neuron_idx = config['neuron_index']
#     hook_timesteps = config["hook_timesteps"]
    
#     # Hook function to modify activations
#     def modify_activation(module, input, output):
#         """
#         Forward hook to zero out the activation coefficient in the output of c_fc for the specified neuron.
#         """
#         print(f"Zeroing out activation coefficient at layer {layer_idx}, neuron {neuron_idx} (output of c_fc)...")
#         print(f"Output shape: {output.shape}")
        
#         with torch.no_grad():
#             # Ensure the neuron index is within the bounds of the output tensor
#             if neuron_idx < output.shape[-1]:
#                 # Zero out the activation coefficient at the specified neuron
#                 output_mod = output.clone()  # Clone the output tensor to modify it
#                 output_mod[:, hook_timesteps, neuron_idx] = 0
#             else:
#                 print(f"Neuron index {neuron_idx} is out of bounds for the output tensor with dimension {output.shape[-1]}")
#                 output_mod = output

#         return output_mod  # Return the modified output tensor

#     # Register the forward hook on the output of the c_fc layer in the specified transformer layer
#     print(f"Registering hook for layer {layer_idx} at neuron {neuron_idx} (output of c_fc)...")
#     hook = model.transformer.h[layer_idx].mlp.c_fc.register_forward_hook(modify_activation)

#     print("Hook registered successfully.")
#     return model, hook  # Return the model and the hook to allow for cleanup later




# def assign_values_to_neurons(model, config):
#     """
#     Modify the activation coefficients for multiple neurons in different layers. Each entry in the config
#     contains a tuple (layer_idx, neuron_idx, assigned_value), and all modifications happen in one hook.
#     """
#     neuron_configs = config['neuron_configs']  # List of (layer_idx, neuron_idx, assigned_value)
#     # hook_timesteps = config["hook_timesteps"]
    
#     # Hook function to modify activations for multiple neurons
#     def modify_activation(module, input, output):
#         """
#         Forward hook to assign specific values to the activation coefficients for the specified neurons.
#         """
#         print(f"Modifying activation values for multiple neurons...")
#         # print(f"Output shape: {output.shape}")
#         # print(f"Input shape: {input[0].shape}")

#         with torch.no_grad():
#             # output_mod = output.clone()  # Clone the output tensor to modify it
#             # input_mod = input[0].clone()

#             # Iterate over the neuron configurations
#             for (layer_idx, neuron_idx, assigned_value) in neuron_configs:
#                 # if neuron_idx < input[0].shape[-1]:
#                 if neuron_idx < output.shape[-1]:
#                     print(f"Assigning value {assigned_value} to neuron {neuron_idx} at layer {layer_idx}...")
#                     # Assign the specific value to the neuron activation
#                     # input[0][:, -1, neuron_idx] = assigned_value
#                     output[:, -1, neuron_idx] = assigned_value
#                 else:
#                     print(f"Neuron index {neuron_idx} is out of bounds for the output tensor with dimension {output.shape[-1]}")

#         return output  # Return the modified output tensor
    
#     # Register a single forward hook on the output of the c_fc layer in the specified transformer layer
#     print(f"Registering hook for modifying multiple neuron activations...")
#     # hook = model.transformer.h[neuron_configs[0][0]].mlp.c_proj.register_forward_hook(modify_activation)
#     hook = model.transformer.h[neuron_configs[0][0]].mlp.c_fc.register_forward_hook(modify_activation)

#     print("Hook registered successfully.")
#     return model, hook  # Return the model and the hook for cleanup later




# def assign_values_to_neurons(model, config):
#     """
#     Modify the activation coefficients for specific neurons in different layers.
#     Each entry in the config contains a tuple (layer_idx, neuron_idx, assigned_value),
#     and one hook is registered per (layer, neuron) pair.
#     """
#     neuron_configs = config['neuron_configs']  # List of (layer_idx, neuron_idx, assigned_value)
#     hook_timesteps = config["hook_timesteps"]

#     # Convert the neuron configuration to GPU tensors for parallelized assignment
#     layer_idxs = torch.tensor([cfg[0] for cfg in neuron_configs], device="cuda")
#     neuron_idxs = torch.tensor([cfg[1] for cfg in neuron_configs], device="cuda")
#     assigned_values = torch.tensor([cfg[2] for cfg in neuron_configs], device="cuda")

#     def patch(layer_idx, neuron_idx_list, assigned_value_list):
#         def hook(module, input, output):
#             """
#             Forward hook to assign specific values to multiple neurons' activation coefficients.
#             """
#             print(f"Modifying activation values for neurons in layer {layer_idx}...")

#             with torch.no_grad():
#                 for neuron_idx, assigned_value in zip(neuron_idx_list, assigned_value_list):
#                     print(f"Assigning value {assigned_value.item()} to neuron {neuron_idx.item()}...")
#                     output[:, hook_timesteps, neuron_idx] = assigned_value

#             return output  # Return the modified pre-GELU activation

#         return hook

#     hooks = []

#     # Group configurations by layer index to enable parallel processing
#     layer_groups = {}
#     for i, layer_idx in enumerate(layer_idxs):
#         if layer_idx.item() not in layer_groups:
#             layer_groups[layer_idx.item()] = ([], [])
#         layer_groups[layer_idx.item()][0].append(neuron_idxs[i])
#         layer_groups[layer_idx.item()][1].append(assigned_values[i])

#     # Register one hook per unique layer, passing multiple neurons to each
#     for layer_idx, (neuron_idx_list, assigned_value_list) in layer_groups.items():
#         neuron_idx_tensor = torch.tensor(neuron_idx_list, device="cuda")
#         assigned_value_tensor = torch.tensor(assigned_value_list, device="cuda")
#         print(f"Registering hook for layer {layer_idx} on {len(neuron_idx_list)} neurons...")

#         hook = model.transformer.h[layer_idx].mlp.c_fc.register_forward_hook(
#             patch(layer_idx, neuron_idx_tensor, assigned_value_tensor)
#         )
#         hooks.append(hook)

#     print(f"Hooks registered successfully for {len(neuron_configs)} neurons across {len(layer_groups)} layers.")
#     return model, hooks  # Return the model and the hooks for cleanup later




def assign_values_to_neurons(model, config):
    """
    Modify the activation coefficients for specific neurons in different layers, with enhanced GPU utilization.
    Each entry in the config contains a tuple (layer_idx, neuron_idx, assigned_value).
    """
    neuron_configs = config['neuron_configs']  # List of (layer_idx, neuron_idx, assigned_value)
    hook_timesteps = config["hook_timesteps"]

    # Convert neuron configs to tensors for batch processing on GPU
    layer_idxs = torch.tensor([cfg[0] for cfg in neuron_configs], device="cuda")
    neuron_idxs = torch.tensor([cfg[1] for cfg in neuron_configs], device="cuda")
    assigned_values = torch.tensor([cfg[2] for cfg in neuron_configs], device="cuda")

    def patch(layer_idx, neuron_idx_tensor, assigned_value_tensor):
        def hook(module, input, output):
            """
            Forward hook to assign specific values to multiple neurons' activation coefficients in parallel.
            """
            with torch.no_grad():
                # Direct batch assignment of values to specified neurons
                output[:, hook_timesteps, neuron_idx_tensor] = assigned_value_tensor
                print(f"Assigned values to neurons in layer {layer_idx}") # {neuron_idx_tensor.tolist()}

            return output  # Return the modified pre-GELU activation

        return hook

    hooks = []

    # Group by layer for parallel assignment within each layer
    layer_groups = {}
    for i, layer_idx in enumerate(layer_idxs):
        if layer_idx.item() not in layer_groups:
            layer_groups[layer_idx.item()] = ([], [])
        layer_groups[layer_idx.item()][0].append(neuron_idxs[i])
        layer_groups[layer_idx.item()][1].append(assigned_values[i])

    # Register one hook per unique layer, assigning multiple neurons in parallel
    for layer_idx, (neuron_idx_list, assigned_value_list) in layer_groups.items():
        neuron_idx_tensor = torch.tensor(neuron_idx_list, device="cuda")
        assigned_value_tensor = torch.tensor(assigned_value_list, device="cuda")

        # Register a single hook for each layer that applies all neuron changes in one batch
        hook = model.transformer.h[layer_idx].mlp.c_fc.register_forward_hook(
            patch(layer_idx, neuron_idx_tensor, assigned_value_tensor)
        )
        hooks.append(hook)

    print(f"Successfully registered hooks for {len(layer_groups)} layers with GPU batch processing.")
    return model, hooks  # Return the model and the hooks for cleanup later



# def assign_values_to_neurons(model, config):
#     """
#     Modify the activation coefficients for specific neurons in different layers. Each entry in the config
#     contains a tuple (layer_idx, neuron_idx, assigned_value), and one hook is registered per (layer, neuron) pair.
#     """
#     neuron_configs = config['neuron_configs']  # List of (layer_idx, neuron_idx, assigned_value)
#     hook_timesteps = config["hook_timesteps"]

#     # Convert the neuron configuration to GPU tensors for parallelized assignment
#     layer_idxs = torch.tensor([cfg[0] for cfg in neuron_configs], device="cuda")
#     neuron_idxs = torch.tensor([cfg[1] for cfg in neuron_configs], device="cuda")
#     assigned_values = torch.tensor([cfg[2] for cfg in neuron_configs], device="cuda")

#     def patch(layer_idx, neuron_idx_list, assigned_value_list):
#         def hook(module, input, output):
#             """
#             Forward hook to assign specific values to multiple neurons' activation coefficients.
#             """
#             print(f"Modifying activation values for neurons in layer {layer_idx}...")

#             with torch.no_grad():
#                 for neuron_idx, assigned_value in zip(neuron_idx_list, assigned_value_list):
#                     if neuron_idx < output.shape[-1]:
#                         print(f"Assigning value {assigned_value.item()} to neuron {neuron_idx.item()}...")
#                         output[:, hook_timesteps, neuron_idx] = assigned_value
#                     else:
#                         print(f"Neuron index {neuron_idx.item()} is out of bounds for the output tensor with dimension {output.shape[-1]}")

#             return output  # Assign the modified pre-GELU activation

#         return hook

#     hooks = []

#     # Group configurations by layer index to enable parallel processing
#     layer_groups = {}
#     for i, layer_idx in enumerate(layer_idxs):
#         if layer_idx.item() not in layer_groups:
#             layer_groups[layer_idx.item()] = ([], [])
#         layer_groups[layer_idx.item()][0].append(neuron_idxs[i])
#         layer_groups[layer_idx.item()][1].append(assigned_values[i])

#     # Register one hook per unique layer, passing multiple neurons to each
#     for layer_idx, (neuron_idx_list, assigned_value_list) in layer_groups.items():
#         neuron_idx_tensor = torch.tensor(neuron_idx_list, device="cuda")
#         assigned_value_tensor = torch.tensor(assigned_value_list, device="cuda")
#         print(f"Registering hook for layer {layer_idx} on {len(neuron_idx_list)} neurons...")

#         hook = model.transformer.h[layer_idx].mlp.c_fc.register_forward_hook(
#             patch(layer_idx, neuron_idx_tensor, assigned_value_tensor)
#         )
#         hooks.append(hook)

#     print(f"Hooks registered successfully for {len(neuron_configs)} neurons across {len(layer_groups)} layers.")
#     return model, hooks  # Return the model and the hooks for cleanup later



def print_and_return_activation_values(model, config):
    """
    Found that output of c_fc is before GeLU, input to c_proj is after GeLU.
    Print dimensions and return the activation values at neuron_idx for the output of c_fc and input of c_proj.
    """
    layer_idx = config['layer_index']
    neuron_idx = config['neuron_index']
    # hook_timesteps = config["hook_timesteps"]

    
    # Hook function to capture and print activations
    def capture_output_to_c_fc(module, input, output):
        """
        Forward hook to print and return dimensions and values at the specific neuron.
        """
        print(f"Capturing activations at layer {layer_idx}, neuron {neuron_idx}...")
        
        # Print dimensions of the output of c_fc
        print(f"Output of c_fc dimensions: {output.shape}")
        
        # Ensure the neuron index is within the bounds of the output tensor
        if neuron_idx < output.shape[-1]:
            neuron_value = output[:, -1, neuron_idx].detach().cpu().numpy() # Take the neuron value at the last time step
            print(f"Output of c_fc at neuron {neuron_idx}: {neuron_value}")
        else:
            print(f"Neuron index {neuron_idx} is out of bounds for c_fc output with dimension {output.shape[-1]}")
            neuron_value = None

        return output  # Return the original output unchanged

    def capture_input_to_c_proj(module, input, output):
        """
        Forward hook to capture the input to c_proj.
        """
        # Print dimensions of the input to c_proj
        print(f"Input to c_proj dimensions: {input[0].shape}")  # Input is a tuple, so input[0] is what we care about

        # Ensure the neuron index is within bounds of the input tensor
        if neuron_idx < input[0].shape[-1]:
            neuron_value = input[0][:, -1, neuron_idx].detach().cpu().numpy() # Take the neuron value at the last time step
            print(f"Input to c_proj at neuron {neuron_idx}: {neuron_value}")
        else:
            print(f"Neuron index {neuron_idx} is out of bounds for c_proj input with dimension {input[0].shape[-1]}")
            neuron_value = None

        return output  # Return the original output unchanged

    # Register the forward hooks on the output of c_fc and the input of c_proj in the specified transformer layer
    print(f"Registering hooks for layer {layer_idx} at neuron {neuron_idx} (c_fc and c_proj)...")
    
    # Hook for c_fc output
    c_fc_hook = model.transformer.h[layer_idx].mlp.c_fc.register_forward_hook(capture_output_to_c_fc)
    
    # Hook for c_proj input
    c_proj_hook = model.transformer.h[layer_idx].mlp.c_proj.register_forward_hook(capture_input_to_c_proj)

    print("Hooks registered successfully.")
    return model, [c_fc_hook, c_proj_hook]  # Return the model and the hooks for later cleanup



# def assign_values_to_neurons(model, config):
#     """
#     Modify the activation coefficients for multiple neurons in different layers. Each entry in the config
#     contains a tuple (layer_idx, neuron_idx, assigned_value), and all modifications happen in one hook.
    
#     Args:
#         model: The transformer model to modify.
#         config: Dictionary containing 'neuron_configs' as a list of tuples (layer_idx, neuron_idx, assigned_value).
    
#     Returns:
#         model: The model with the hook registered.
#         hooks: A list of hook handles for cleanup later.
#     """
#     neuron_configs = config['neuron_configs']  # List of (layer_idx, neuron_idx, assigned_value)
#     hooks = []  # List to store hook handles for cleanup
    
#     # Define hook function to modify activations
#     def modify_activation(module, input, output):
#         """
#         Forward hook to assign specific values to the activation coefficients for the specified neurons.
#         """
#         print(f"Modifying activation values for multiple neurons...")
#         print(f"Output shape: {output.shape}")

#         with torch.no_grad():
#             output_mod = output.clone()  # Clone the output tensor to modify it

#             # Iterate over the neuron configurations
#             for (layer_idx, neuron_idx, assigned_value) in neuron_configs:
#                 if neuron_idx < output.shape[-1]:
#                     print(f"Assigning value {assigned_value} to neuron {neuron_idx} at layer {layer_idx}...")
#                     # Assign the specific value to the neuron activation
#                     output_mod[:, -1, neuron_idx] = assigned_value
#                 else:
#                     print(f"Neuron index {neuron_idx} is out of bounds for the output tensor with dimension {output.shape[-1]}")

#         return output_mod  # Return the modified output tensor

#     # Register a hook for each layer in neuron_configs
#     for (layer_idx, neuron_idx, _) in neuron_configs:
#         print(f"Registering hook for layer {layer_idx} and neuron {neuron_idx}...")
#         hook = model.blocks[layer_idx].mlp.hook_post.register_forward_hook(modify_activation)
#         hooks.append(hook)

#     print("Hooks registered successfully.")
#     return model, hooks  # Return the model and list of hooks for cleanup later



           