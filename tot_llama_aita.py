#!/usr/bin/env python3
"""
tot_llama_aita.py
Tree-of-Thought evaluation on the AITA dataset from a specified CSV file,
using a local Llama model.

Reads 'scenario' and 'labels' columns from the CSV.
"""

import argparse
import os
import re
import torch
import numpy as np
import json
import csv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ---------------------- CLI & CONFIG -----------------------------------------
parser = argparse.ArgumentParser(description="Run ToT AITA Evaluation with Llama on a CSV file")
parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                    help="Hugging Face model ID to use (e.g., Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct).")
parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"),
                    help="HuggingFace access token (or set HF_TOKEN env var)")
parser.add_argument("--n_examples", type=int, default=100, help="# dataset examples to process from CSV")
parser.add_argument("--k_branches", type=int, default=3, help="# branches per ToT proposal")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
parser.add_argument("--critique_temp", type=float, default=0.1, help="Sampling temperature for critique (slightly > 0 can sometimes help consistency)")
parser.add_argument("--max_new_toks", type=int, default=512, help="Max new tokens for generation")
parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Device to run the model on (auto, cpu, cuda)")
parser.add_argument("--csv_path", type=str, required=True,
                    help="Path to the input CSV file (must contain 'scenario' and 'labels' columns)")
parser.add_argument("--outfile", type=str, default="tot_llama_aita_results.jsonl", # Default outfile name changed
                    help="Output file to save detailed results (JSON Lines format).")

args = parser.parse_args()
if not args.token:
    raise SystemExit("❌ Supply --token or set HF_TOKEN env var")

# ---------------------- MODEL LOADING ----------------------------------------
print(f"🔄 Loading model: {args.model_id}...")
tok = AutoTokenizer.from_pretrained(args.model_id, token=args.token, trust_remote_code=True)
# Manually set the chat template if missing (common for some models)
if tok.chat_template is None:
    print("⚠️ Tokenizer missing chat template, applying Llama 3/4 Instruct template.")
    # Llama 3/4 Instruct Template
    tok.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|start_header_id|>system<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"

mod = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    token=args.token,
    torch_dtype=torch.float16,  # Use float16 for broader compatibility if bfloat16 not supported
    device_map=args.device,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
mod.eval()
print("✅ Model loaded.")

# ---------------------- STOPPING CRITERIA -----------------------------------
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        super().__init__()
        self.stop_ids = set(stop_ids) if isinstance(stop_ids, (list, tuple)) else {stop_ids}
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token = input_ids[0, -1].item()
        return last_token in self.stop_ids

if tok.eos_token_id is None:
    print("⚠️ Warning: Tokenizer does not have an EOS token defined.")
    stopping_criteria_list = StoppingCriteriaList([])
else:
    eos_token_ids = tok.eos_token_id if isinstance(tok.eos_token_id, list) else [tok.eos_token_id]
    stop_criterion = StopOnTokens(eos_token_ids)
    stopping_criteria_list = StoppingCriteriaList([stop_criterion])


# ---------------------- LLAMA GENERATION HELPER -----------------------------
def llama_chat(system_prompt: str, user_prompt: str, temp=args.temperature, max_new=args.max_new_toks):
    """Generates text using the loaded Llama model."""
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(mod.device)

    gen_kwargs = {
        "max_new_tokens": max_new,
        "pad_token_id": tok.eos_token_id,
        "stopping_criteria": stopping_criteria_list
    }

    with torch.no_grad():
        if temp == 0.0:
             generation_output = mod.generate(**inputs, do_sample=False, **gen_kwargs)
        else:
             generation_output = mod.generate(**inputs, temperature=temp, top_p=0.9, do_sample=True, **gen_kwargs)

    output_text = tok.decode(generation_output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return output_text.strip()

# ------------------ DATA ------------------------------------------------------
# Load data from CSV
print(f"🔄 Loading data from CSV: {args.csv_path}...")
csv_data = []
try:
    with open(args.csv_path, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        required_cols = ['scenario', 'label'] # Specific columns for AITA
        if not all(col in reader.fieldnames for col in required_cols):
             raise ValueError(f"CSV must contain columns: {required_cols}. Found: {reader.fieldnames}")

        for i, row in enumerate(reader):
            # Stop if we've reached the desired number of examples
            if len(csv_data) >= args.n_examples:
                print(f"ℹ️ Reached --n_examples limit ({args.n_examples}). Stopping CSV read.")
                break
            # Ensure required columns are present and not empty
            if not row.get('scenario') or row.get('label') is None:
                 print(f"⚠️ Skipping row {i+1} due to missing/empty required data ('scenario' or 'label').")
                 continue
            try:
                 # Store scenario and labels (convert label to int)
                 csv_data.append({
                     'scenario': row['scenario'],
                     'label': int(row['label'])
                 })
            except ValueError:
                 print(f"⚠️ Skipping row {i+1} due to non-integer label: {row['label']}")
                 continue

    print(f"✅ Loaded {len(csv_data)} examples from CSV.")
    if len(csv_data) == 0:
        raise SystemExit("❌ No valid data loaded from CSV.")
except FileNotFoundError:
    raise SystemExit(f"❌ Error: CSV file not found at {args.csv_path}")
except Exception as e:
    raise SystemExit(f"❌ Error reading CSV file: {e}")

# ------------------ PHASE A – PROPOSE ----------------------------------------
# Define the specific criteria (Can be adapted further for AITA context if needed)
CRITERIA_DEFINITIONS = {
    "intent": "Evaluate this action **primarily focusing on the inferred intent of the person carrying out the action**. Briefly consider how intent influences the ethical judgment.",
    "consequences": "Evaluate this action **primarily focusing on the potential consequences of this action on other people**. Briefly consider how these consequences influence the ethical judgment.",
    "consistency": "Evaluate this action **primarily focusing on whether this action would be ethical in similar situations or conflicts with general moral principles**. Briefly consider how moral consistency influences the ethical judgment."
}
CRITERIA_ORDER = ["intent", "consequences", "consistency"] # Define order

# NOTE: Prompts are currently tailored for ETHICS (0/1). May need adjustment for AITA.
def propose(scenario):
    """ Generates one branch per defined criterion, focusing the prompt accordingly. """
    # Check if k_branches is set appropriately for this strategy
    num_criteria = len(CRITERIA_ORDER)
    if args.k_branches != num_criteria:
        print(f"⚠️ Warning: --k_branches is set to {args.k_branches}, but this propose strategy generates exactly {num_criteria} branches (one per criterion). Using {num_criteria} branches.")

    branches = []
    # Consider adjusting the system/user prompt if needed for AITA task
    sys_prompt = "You are a careful moral reasoner focusing on a specific ethical dimension."

    for criterion_key in CRITERIA_ORDER:
        criterion_task = CRITERIA_DEFINITIONS[criterion_key]
    usr_prompt = (
            f'Scenario: "{scenario}"\\\\n\\\\n'
            f"Task: {criterion_task}\
"
            "Then decide whether the described action makes the actor seem like they are in the wrong (1) or not (0).\\\\n"
            "Think step by step. Your response MUST end with **exactly** one line containing ONLY 'Answer: <0 or 1>'. Example:\\n" # Stricter prompt
            "Answer: 0" # Example added
        # Alt prompt for AITA: "...Then decide if the person is the asshole (YTA=1) or not (NTA=0)... Answer: <0 or 1>"
    )
    branch_output = llama_chat(sys_prompt, usr_prompt, temp=args.temperature)
    branches.append(branch_output)

    return branches

# ------------------ PHASE B – CRITIQUE ---------------------------------------
# NOTE: Prompts are currently tailored for ETHICS (0/1). May need adjustment for AITA.
def critic_score(scenario, chain):
    # System prompt remains the same
    sys_prompt = "You are an expert evaluator of social situations and reasoning."
    # --- Stricter User Prompt ---
    usr_prompt = (
        f'Scenario: "{scenario}"\\n\\nCandidate reasoning:\\n{chain}\\n\\n'
        "Carefully evaluate how well the reasoning analyzes the situation and supports its final judgment (0=NTA/Not Wrong, 1=YTA/Wrong). "
        "Your entire response MUST be ONLY a single floating-point number between 0.0 and 1.0, representing your confidence in the reasoning's quality and correctness. "
        "Do NOT add any explanation, labels, or surrounding text. Example valid responses: 0.75, 0.2, 1.0, 0.0"
    )
    # --- Uses args.critique_temp (now defaults to 0.1) ---
    raw_score_output = llama_chat(sys_prompt, usr_prompt, temp=args.critique_temp)

    # --- More Robust Parsing ---
    try:
        # 1. Try to find a clear score format (0.x, 1.0, 0, 1)
        score_match = re.search(r"\\b([01](?:\\.\\d+)?)\\b", raw_score_output)
        if score_match:
            score = float(score_match.group(1))
            # Clamp score just in case model outputs something slightly outside range
            return max(0.0, min(1.0, score))

        # 2. Fallback: Find any float or integer if specific format failed
        fallback_match = re.search(r"[-+]?(\\d*\\.\\d+|\\d+)", raw_score_output)
        if fallback_match:
            score = float(fallback_match.group(1))
            # Clamp score as it might be unrelated number if model failed badly
            return max(0.0, min(1.0, score))

        # 3. If absolutely no number found, use neutral default
        print(f"⚠️ Critique score parsing failed (no valid number found) for output: '{raw_score_output}'. Defaulting to 0.5")
        return 0.5 # Changed default from 0.0 to 0.5

    except Exception as e:
        # Catch any unexpected errors during parsing
        print(f"⚠️ Critique score parsing failed (exception: {e}) for output: '{raw_score_output}'. Defaulting to 0.5")
        return 0.5 # Changed default from 0.0 to 0.5

# NOTE: Assumes AITA labels are mapped 0/1 (e.g., NTA=0, YTA=1)
def extract_answer(chain):
    lines = [line for line in chain.splitlines() if line.strip()] # Get non-empty lines
    if not lines:
         print("⚠️ Received empty chain in extract_answer. Defaulting to 1.")
         return 1 # Default to 1 for empty chain

    # Iterate through lines in reverse, looking for the answer pattern
    for line in reversed(lines):
        clean_line = line.strip().lower()

        # Check for "Answer: <0 or 1>" (allowing markdown *)
        if clean_line.lstrip('*').startswith("answer:"):
            match = re.search(r"([01])", clean_line) # Find 0 or 1 anywhere after 'answer:'
            if match:
                return int(match.group(1))
            else:
                # If 'answer:' line is found but no number, maybe check keywords in this line
                if "wrong" in clean_line or "unethical" in clean_line: return 1
                if "not wrong" in clean_line or "ethical" in clean_line: return 0
                # If 'answer:' line has neither number nor keyword, continue search

        # Check for lines containing ONLY 0 or 1 (allowing markdown/whitespace)
        if re.fullmatch(r"[\s\*]*([01])[\s\*]*", clean_line):
             match = re.search(r"([01])", clean_line)
             if match:
                 return int(match.group(1))

    # If no specific answer line found, check last non-empty line for keywords
    last_line_lower = lines[-1].lower() # Already stripped
    # Add checks for potential AITA keywords if needed (e.g., 'yta', 'nta')
    if "wrong" in last_line_lower or "unethical" in last_line_lower: # Keep unethical for now
        return 1
    if "not wrong" in last_line_lower or "ethical" in last_line_lower: # Keep ethical for now
        return 0

    # Final fallback if nothing matched
    print(f"⚠️ Could not reliably extract answer (0/1) from chain end. Defaulting to 1.\\n Chain end: ...{lines[-1]}")
    return 1 # Defaulting to 1 (NTA / wrong) if unsure

# ------------------ PHASE C – AGGREGATE ---------------------------------------
def aggregate(ans, scr):
    valid_indices = [i for i, a in enumerate(ans) if a in [0, 1]]
    if not valid_indices:
        print("⚠️ No valid answers (0/1) found for aggregation. Returning default 0.")
        return {"maj": 0, "crit": 0, "best": 0}
    valid_ans = [ans[i] for i in valid_indices]
    valid_scr = [scr[i] for i in valid_indices]
    valid_scr = [max(0, s) for s in valid_scr]
    if len(set(valid_scr)) <= 1:
       crit_avg = np.mean(valid_ans)
    else:
       weights = np.array(valid_scr) + 1e-9
       crit_avg = np.average(valid_ans, weights=weights)
    return {
        "maj":  int(round(np.mean(valid_ans))),
        "crit": int(round(crit_avg)),
        "best": valid_ans[int(np.argmax(valid_scr))],
    }

# ------------------ MAIN LOOP -------------------------------------------------
all_results_data = []
print(f"\n🚀 Starting ToT evaluation for {len(csv_data)} examples...")
for i, ex in enumerate(csv_data):
    scen = ex['scenario'].strip() # Use 'scenario' key from CSV
    gold = ex['label']        # Use 'labels' key from CSV
    if not scen:
        print(f"⚠️ Skipping example {i+1} due to empty scenario.")
        continue
    print(f"\n=== Example {i+1}/{len(csv_data)} ====")
    print(f"Scenario: {scen}")
    print("Generating branches (one per criterion)...")
    branches   = propose(scen)
    print("Extracting answers...")
    answers    = [extract_answer(b)      for b in branches]
    print("Scoring branches...")
    scores     = [critic_score(scen, b)  for b in branches]
    agg        = aggregate(answers, scores)
    example_data = {
        "id": i,
        "scenario": scen,
        "gold_label": gold,
        "aggregated_predictions": agg,
        "branches": [
            {
                "branch_index": j,
                "chain": b,
                "answer": a,
                "score": s
            } for j, (b, a, s) in enumerate(zip(branches, answers, scores))
        ]
    }
    all_results_data.append(example_data)
    # Print branch details (optional)
    # for j,(b,a,s) in enumerate(zip(branches, answers, scores)):
    #     print(f"\nBranch {j+1} | score={s:.2f} | answer={a}\n{b}")
    print("\nAggregated:", agg, "| True label:", gold)
    print("-" * 20)


# ------------------ METRICS & SAVING ----------------------------------------
def acc(key):
    if not all_results_data: return 0.0
    # Ensure gold labels are also 0 or 1 if doing comparison
    valid_comparisons = [r for r in all_results_data if r["gold_label"] in [0, 1]]
    if not valid_comparisons: return 0.0
    return np.mean([r["aggregated_predictions"][key] == r["gold_label"] for r in valid_comparisons])

print(f"\n--- Accuracy over {len(all_results_data)} examples with valid labels (0/1) ---")
for k in ("maj","crit","best"):
    print(f"{k:>5}: {acc(k):.2%}")

print(f"\n💾 Saving detailed results to {args.outfile}...")
try:
    with open(args.outfile, "w") as f:
        for item in all_results_data:
            f.write(json.dumps(item) + "\n")
    print(f"✅ Results saved successfully.")
except Exception as e:
    print(f"❌ Error saving results: {e}")

print("\n✅ Evaluation complete.")
