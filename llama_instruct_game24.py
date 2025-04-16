import os
import sys
import argparse
import time 
import io

# --- Output Redirection Setup ---
log_file_path = "llama_instruct_game24_output.log"
original_stdout = sys.stdout
original_stderr = sys.stderr
log_file = None

class Tee(io.TextIOBase):
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)
        self.stream1.flush()
        self.stream2.flush()
    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

try:
    log_file = open(log_file_path, 'w')
    sys.stdout = Tee(log_file, original_stdout)
    sys.stderr = Tee(log_file, original_stderr)
    print(f"--- Script started. Output is being mirrored to {log_file_path} ---")

    # Set Hugging Face token 
    hf_token = "YOUR_TOKEN_HERE"
    os.environ["HF_TOKEN"] = hf_token
    print(f"{time.time():.2f} - HF Token set.")

    # Add the src directory to Python path
    sys.path.insert(0, os.path.abspath('src'))
    print(f"{time.time():.2f} - src added to path.")

    # --- Import necessary components ---
    try:
        from tot.methods.bfs import solve
        from tot.models import gpt 
        from tot.tasks.game24 import Game24Task
        print(f"{time.time():.2f} - Imports successful.")
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during import: {e}")
        sys.exit(1)

    # --- Use Synthetic Game Data --- 
    game_data = ['1 2 3 4', '5 6 7 8', '9 8 7 6'] # Simple inline data
    print(f"{time.time():.2f} - Using synthetic game data: {game_data}")

    # --- Configure Arguments --- 
    args = argparse.Namespace(
        backend='meta-llama/Llama-3.2-3B-Instruct',
        temperature=0.7,
        task='game24',
        naive_run=False,
        prompt_sample=None, 
        method_generate='propose',
        method_evaluate='value',
        method_select='greedy',
        n_generate_sample=1,
        n_evaluate_sample=3,
        n_select_sample=5
    )
    print(f"Using backend: {args.backend}")

    # Initialize task with synthetic data
    task = Game24Task(data=game_data)

    # Get first puzzle input from our synthetic data
    idx = 0 
    input_numbers = task.get_input(idx)
    print(f"Starting Game 24 solve for puzzle #{idx}: {input_numbers}")
    
    # --- Run the solve process --- 
    print(f"\n{time.time():.2f} - Calling solve function...")
    start_solve_time = time.time()

    try:
        ys, infos = solve(args, task, idx, to_print=True)
        end_solve_time = time.time()
        print(f"{end_solve_time:.2f} - Solve function finished. Duration: {end_solve_time - start_solve_time:.2f} seconds.")

        # Print the final solution(s)
        print("\nFinal Solution(s):")
        if ys:
            for solution in ys:
                print(solution)
        else:
            print("No solution found")
        
    except Exception as e:
        end_solve_time = time.time()
        print(f"{end_solve_time:.2f} - ERROR during solve: {e}")

    print(f"\n{time.time():.2f} - Main script logic finished.")

finally:
    # --- Restore Output Streams and Close File ---
    if isinstance(sys.stdout, Tee):
        sys.stdout = original_stdout
    if isinstance(sys.stderr, Tee):
        sys.stderr = original_stderr
    if log_file:
        log_file.close()
    print(f"--- Script finished. Output mirroring stopped. Log saved to {log_file_path} ---") 
