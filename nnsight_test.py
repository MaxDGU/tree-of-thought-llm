from nnsight import CONFIG, LanguageModel
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


llm = LanguageModel("meta-llama/Llama-3.2-1B", device_map="auto")
# Keep your existing API configuration
CONFIG.set_default_api_key("66f3698257fb499b97bff0e5c79532e8") #nnsight api key
os.environ["HF_TOKEN"] = "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" #huggingface api key

with llm.trace("Hello, how are you?") as tracer:
    hidden_states = llm.model.layers[-1].output.save()
    output = llm.output.save()

'''
NNSIGHT REMOTE EXECUTION
# All we need to specify using NDIF vs executing locally is remote=True.
with llama.trace("The Eiffel Tower is in the city of", remote=True) as runner:

    hidden_states = llama.model.layers[-1].output.save()

    output = llama.output.save()

print(hidden_states)

print(output["logits"])
'''

print(hidden_states)
print(output)
