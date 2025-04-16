import torch
from nnsight import LanguageModel
from src.tot.models import load_model  # We might use this for tokenizer or compare behavior
import os

from nnsight import CONFIG

# Configure nnsight with your API key
# Replace "YOUR_API_KEY" with the actual key
CONFIG.set_default_api_key("66f3698257fb499b97bff0e5c79532e8")
print("NNsight API key configured.")

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Or your desired Llama model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Loading model {MODEL_NAME} with nnsight...")

# Load the model using nnsight
# We need to ensure it uses the same arguments (like low_cpu_mem_usage if applicable)
# as our original load_model for consistency, if possible.
# nnsight might handle device_map differently, often loading to meta first.
try:
    # Use LanguageModel instead of Llama
    model = LanguageModel(MODEL_NAME, device_map=DEVICE, torch_dtype=torch.float16)
    print(f"Model {MODEL_NAME} loaded successfully via nnsight.")
    # Try getting the tokenizer from nnsight's wrapper if possible
    # or load it separately if needed.
    # tokenizer = model.tokenizer # Common pattern in nnsight examples
    # If the above doesn't work, load tokenizer separately:
    from transformers import AutoTokenizer
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, token=hf_token)
    print("Tokenizer loaded separately.")

except Exception as e:
    print(f"Error loading model/tokenizer with nnsight: {e}")
    print("Attempting to load tokenizer separately if model loading failed partially...")
    try:
        from transformers import AutoTokenizer
        hf_token = os.getenv("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, token=hf_token)
        print("Tokenizer loaded separately.")
    except Exception as te:
        print(f"Error loading tokenizer separately: {te}")
        tokenizer = None
    model = None # Ensure model is None if loading failed

if model and tokenizer:
    print("Model and tokenizer ready.")
    # Basic test generation (optional)
    # prompt = "The game of 24 input is 4 5 6 7. The first step is"
    # with model.generate(max_new_tokens=10) as generator:
    #     with generator.invoke(prompt) as invoker:
    #         pass
    # output = model.tokenizer.decode(generator.output[0])
    # print(f"Test generation output: {output}")

else:
    print("Failed to load model or tokenizer. Exiting.")
    exit()

# Define prompts for the Game of 24
# Example using numbers 4, 5, 6, 7 -> Goal 24
# Potential solution: (6 + 5 - 7) * 4 = (11 - 7) * 4 = 4 * 4 = 16 (Incorrect example, need a good one)
# Actual solution: (4 + 6) * (7 - 5) = 10 * 2 = 20 (Incorrect)
# Actual solution: 6 * 5 + 7 - 4 = 30 + 7 - 4 = 33 (Incorrect)
# Actual solution: 4 * (6 + 7 - 5) = 4 * (13 - 5) = 4 * 8 = 32 (Incorrect)
# Actual solution: 6 * (5 + 7/4) NO
# Actual solution: (6+4)*(7-5) = 10*2 = 20 NO
# Actual solution: (5 + 7) * (6-4) = 12 * 2 = 24 YES!

source_prompt = "Input: 4 5 6 7. Progress: 5 + 7 = 12. Next expression:" # Part of a known solution
target_prompt = "Input: 4 5 6 7. Progress: 4 + 5 = 9. Next expression:" # A different, potentially less direct path

# Define the layer and token position for patching
layer_to_patch = len(model.model.layers) - 1  # Last layer index

print(f"Patching activation: Layer {layer_to_patch}, Last Token")
# Reduce tokens for simpler test
MAX_NEW_TOKENS = 1 

# Placeholders for the saved proxies
source_output_proxy = None
source_activation = None
original_target_output_proxy = None
patched_target_output_proxy = None

# --- Run 1: Source Prompt --- #
print(f"\n--- Running Source Prompt (Remote) ---")
print(f"Prompt: {source_prompt}")

with model.generate(max_new_tokens=MAX_NEW_TOKENS, remote=True, scan=False, validate=False) as generator:
    with generator.invoke(source_prompt) as invoker:
        # Access the activation location - MLP output
        activation_proxy = model.model.layers[layer_to_patch].mlp.output
        # Save the activation for the LAST token of the prompt
        source_activation = activation_proxy[:, -1, :].save()
        # Save the output proxy inside invoke
        source_output_proxy = model.output.save()

# Check if proxies were saved before accessing
if source_output_proxy is not None and source_activation is not None:
    output_value = source_output_proxy # Get the proxy value/object (should be a dict)
    
    # Attempt decoding based on inspection - using 'sequences' key
    source_output = "[Decoding Failed]"
    try:
        if isinstance(output_value, dict) and 'sequences' in output_value:
            source_output_tokens = output_value['sequences']
            source_output = model.tokenizer.decode(source_output_tokens[0])
        else:
            print(f"WARN: Could not find 'sequences' key in source output object of type {type(output_value)}")
    except Exception as e:
        print(f"ERROR during source decoding attempt: {e}")

    # Access activation shape directly (no .value)
    print(f"Source Activation Shape (Last Token): {source_activation.shape}")
    print(f"Source Output: {source_output}")
else:
    print("ERROR: Failed to save source activation or output proxy.")
    exit()

# --- Run 2: Target Prompt (Original - No Patching) --- #
print(f"\n--- Running Target Prompt (Original, Remote) ---")
print(f"Prompt: {target_prompt}")

with model.generate(max_new_tokens=MAX_NEW_TOKENS, remote=True, scan=False, validate=False) as generator:
    with generator.invoke(target_prompt) as invoker:
        # Save the output proxy inside invoke
        original_target_output_proxy = model.output.save()

if original_target_output_proxy is not None:
    output_value = original_target_output_proxy
    original_target_output = "[Decoding Failed]"
    try:
        if isinstance(output_value, dict) and 'sequences' in output_value:
            original_target_output_tokens = output_value['sequences']
            original_target_output = model.tokenizer.decode(original_target_output_tokens[0])
        else:
            print(f"WARN: Could not find 'sequences' key in original target output object of type {type(output_value)}")
    except Exception as e:
        print(f"ERROR during original target decoding: {e}")
    print(f"Target Output (Original): {original_target_output}")
else:
    print("ERROR: Failed to save original target output proxy.")
    exit()

# --- Run 3: Target Prompt (Patched) --- #
print(f"\n--- Running Target Prompt (Patched, Remote) ---")
print(f"Prompt: {target_prompt}")

with model.generate(max_new_tokens=MAX_NEW_TOKENS, remote=True, scan=False, validate=False) as generator:
    with generator.invoke(target_prompt) as invoker:
        # Target the same MLP output proxy
        activation_proxy = model.model.layers[layer_to_patch].mlp.output
        if source_activation is not None:
            # Edit the unsliced MLP output proxy
            activation_proxy.edit(source_activation)
        else:
            print("WARN: source_activation was None, skipping patch.")
        # Save the output proxy inside invoke
        patched_target_output_proxy = model.output.save()

if patched_target_output_proxy is not None:
    output_value = patched_target_output_proxy
    patched_target_output = "[Decoding Failed]"
    try:
        if isinstance(output_value, dict) and 'sequences' in output_value:
            patched_target_output_tokens = output_value['sequences']
            patched_target_output = model.tokenizer.decode(patched_target_output_tokens[0])
        else:
            print(f"WARN: Could not find 'sequences' key in patched target output object of type {type(output_value)}")
    except Exception as e:
        print(f"ERROR during patched target decoding: {e}")
    print(f"Target Output (Patched): {patched_target_output}")
else:
    print("ERROR: Failed to save patched target output proxy.")
    exit()

# Compare outputs
print("\n--- Comparison ---")
print(f"Source Output:        {source_output}")
print(f"Target Output (Orig): {original_target_output}")
print(f"Target Output (Patch):{patched_target_output}")

# --- Next steps: Integrate with ToT logic, try different layers/activations, compare ToT vs I/O activations --- 