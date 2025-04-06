import os
import json
import random
import shutil

import pandas as pd

from collections import defaultdict


def process_outputs_folder(folder: str, processor_fn) -> dict:
    """
    Common helper function to process train/test output folders.
    
    Args:
        folder: Path to the base results folder containing args.json
        processor_fn: Function that processes a single folder's statistics
        
    Returns:
        Dict with 'train' and/or 'test' keys mapping to processor_fn results
    """
    # Read args.json for configuration
    args_path = os.path.join(folder, "args.json")
    with open(args_path, 'r') as f:
        args = json.load(f)
    
    max_attempts = args.get('max_attempts_per_task', 1)
    results = {}
    
    # Process train and test separately if their flags are True
    if args.get('do_train', False):
        train_folder = os.path.join(folder, "train_outputs")
        results['train'] = processor_fn(train_folder, max_attempts)
        
    if args.get('do_test', False):
        test_folder = os.path.join(folder, "test_outputs")
        results['test'] = processor_fn(test_folder, max_attempts)
    
    if not results:
        raise ValueError("Neither do_train nor do_test is True in args.json")
        
    return results


def get_successes_by_attempts(folder: str) -> dict:
    """
    Analyzes success rates across multiple attempts for tasks.
    
    Args:
        folder: Path to the base results folder containing args.json
        
    Returns:
        Dict with keys 'train' and/or 'test' (depending on flags), where each value is a dict containing:
        - 'successes_by_attempt': Dict mapping attempt number to number of successes
        - 'tasks_by_attempt': Dict mapping attempt number to total tasks attempted
        - 'cumulative_successes': List of cumulative successes by attempt number
        - 'total_tasks': Total number of unique task IDs
    """
    return process_outputs_folder(folder, process_folder_stats)


def process_folder_stats(base_folder: str, max_attempts: int) -> dict:
    """Helper function to process statistics for a single folder."""
    successes_by_attempt = defaultdict(int)
    tasks_by_attempt = defaultdict(int)
    total_tasks = 0
    
    # Iterate through task folders
    for task_id in os.listdir(base_folder):
        task_path = os.path.join(base_folder, task_id)
        if not os.path.isdir(task_path):
            continue
            
        total_tasks += 1
        
        # Check each attempt file
        for i in range(max_attempts):
            output_file = os.path.join(task_path, f"output_{i}.json")
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                        if data.get('reward', False):
                            successes_by_attempt[i] += 1
                        tasks_by_attempt[i] += 1
                except (json.JSONDecodeError, FileNotFoundError):
                    continue

    # Calculate cumulative successes
    cumulative_successes = []
    cumulative_success_rate = []
    running_sum = 0
    
    for i in range(max_attempts):
        running_sum += successes_by_attempt[i]
        cumulative_successes.append(running_sum)
        cumulative_success_rate.append(running_sum / total_tasks)

    return {
        'successes_by_attempt': dict(successes_by_attempt),
        'tasks_by_attempt': dict(tasks_by_attempt),
        'cumulative_successes': cumulative_successes,
        'cumulative_success_rate': cumulative_success_rate,
        'total_tasks': total_tasks
    }

def sample_status_differences(
    csv_path: str,
    exp1_name: str,
    exp2_name: str,
    model_name: str,
    output_base_dir: str,
    sample_size: int = 10,
    random_seed: int = 42
) -> dict:
    """
    Saves log files for tasks that have different statuses between two experiments.
    Done at random and sampled
    
    Args:
        csv_path: Path to CSV containing status differences
        exp1_name: Name of first experiment
        exp2_name: Name of second experiment  
        model_name: Name of the model
        output_base_dir: Base directory to save outputs
        sample_size: Maximum number of tasks to sample per category
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict containing:
        - 'wrong_to_right': List of sampled task IDs where exp1 failed but exp2 passed
        - 'right_to_wrong': List of sampled task IDs where exp1 passed but exp2 failed
    """
    # Set random seed
    random.seed(random_seed)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Get column names from CSV (they should end with _status)
    status_cols = [col for col in df.columns if col.endswith('_status')]
    if len(status_cols) != 2:
        raise ValueError(f"Expected 2 status columns, found {len(status_cols)}")
    
    # Group tasks
    wrong_to_right = df[
        (df[status_cols[0]] == 'fail') & 
        (df[status_cols[1]] == 'pass')
    ]['task_id'].tolist()
    
    right_to_wrong = df[
        (df[status_cols[0]] == 'pass') & 
        (df[status_cols[1]] == 'fail')
    ]['task_id'].tolist()
    
    # Sample tasks
    wrong_to_right = random.sample(wrong_to_right, min(sample_size, len(wrong_to_right)))
    right_to_wrong = random.sample(right_to_wrong, min(sample_size, len(right_to_wrong)))
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, f"{exp1_name}_{exp2_name}_{model_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy log files for each task
    for task_id in wrong_to_right + right_to_wrong:
        # Sanitize task_id for file operations by replacing slashes with underscores
        safe_task_id = task_id.replace('/', '_')
        
        # Determine status for folder name
        if task_id in wrong_to_right:
            status_dir = f"fail_pass_{safe_task_id}"
        else:
            status_dir = f"pass_fail_{safe_task_id}"
            
        task_dir = os.path.join(output_dir, status_dir)
        os.makedirs(task_dir, exist_ok=True)
        
        # Copy log files from both experiments
        for exp_name, status in [(exp1_name, status_cols[0]), (exp2_name, status_cols[1])]:
            # Use original task_id for source path since that's the actual file structure
            src_log = os.path.join("final_results", f"{exp_name}_{model_name}", 
                                 "test_outputs", safe_task_id, "logfile.log")
            dst_log = os.path.join(task_dir, f"{exp_name}.log")
            
            if os.path.exists(src_log):
                shutil.copy2(src_log, dst_log)
            else:
                print(f"Warning: Log file not found for {task_id} in {exp_name}")


def get_skill_library_usage(folder: str) -> dict:
    """
    Analyzes skill library usage across task attempts.
    
    Args:
        folder: Path to the base results folder containing args.json
        
    Returns:
        Dict with keys 'train' and/or 'test' (depending on flags), where each value is a dict containing:
        - 'used_in_any_attempt': Number of tasks that used skill library in at least one attempt
        - 'used_in_last_attempt': Number of tasks that used skill library in their last attempt
        - 'total_tasks': Total number of tasks
    """
    return process_outputs_folder(folder, process_skill_usage)


def process_skill_usage(base_folder: str, max_attempts: int) -> dict:
    """Helper function to process skill library usage for a single folder."""
    used_in_any = 0
    used_in_last = 0
    total_tasks = 0
    used_in_any_tasks = []
    used_in_last_tasks = []
    
    # Iterate through task folders
    for task_id in os.listdir(base_folder):
        task_path = os.path.join(base_folder, task_id)
        if not os.path.isdir(task_path):
            continue
            
        total_tasks += 1
        any_used = False
        last_attempt_used = False
        
        # Check each attempt file
        for i in range(max_attempts):
            output_file = os.path.join(task_path, f"output_{i}.json")
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                        last_attempt_used = data.get('dependency_used', False)
                        if last_attempt_used:
                            any_used = True
                except (json.JSONDecodeError, FileNotFoundError):
                    continue
        
        if any_used:
            used_in_any += 1
            used_in_any_tasks.append(task_id)
        if last_attempt_used:
            used_in_last += 1
            used_in_last_tasks.append(task_id)

    return {
        'used_in_any_attempt': used_in_any,
        'used_in_any_attempt_tasks': used_in_any_tasks,
        'used_in_last_attempt': used_in_last,
        'used_in_last_attempt_tasks': used_in_last_tasks,
        'total_tasks': total_tasks
    }

def print_failed_tasks(base_folder: str):
    """
    Prints task IDs for tasks where the reward in output.json is False.
    
    Args:
        base_folder: Path to the folder containing task_id folders
    """
    failed_count = 0
    
    # Iterate through task folders
    for task_id in os.listdir(base_folder):
        task_path = os.path.join(base_folder, task_id)
        if not os.path.isdir(task_path):
            continue
            
        # Check output.json
        output_file = os.path.join(task_path, "output.json")
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    if not data.get('reward', False):
                        print(f"{failed_count}. Failed task: {task_id}")
                        failed_count += 1
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Error reading output for task: {task_id}")
                
    print(f"\nTotal failed tasks: {failed_count}")

def get_train_stats_by_iter(folder: str) -> dict:
    """
    Analyzes training success rates by iteration order.
    
    Args:
        folder: Path to the base results folder containing args.json
        
    Returns:
        Dict with training statistics including:
        - 'successes_by_iter': Dict mapping iteration number to success (1 or 0)
        - 'total_tasks': Total number of tasks
    """
    return process_outputs_folder(folder, process_train_stats)

def process_train_stats(base_folder: str, max_attempts: int) -> dict:
    """Helper function to process training statistics with iteration ordering."""
    successes_by_iter = {}
    total_tasks = 0
    
    # Iterate through task folders
    for task_id in os.listdir(base_folder):
        task_path = os.path.join(base_folder, task_id)
        if not os.path.isdir(task_path):
            continue
            
        # Print progress every 50 tasks
        if total_tasks % 50 == 0:
            print(f"Progress: {total_tasks} tasks processed")
            
        total_tasks += 1
        
        # Get iteration number from logfile
        logfile_path = os.path.join(task_path, "logfile.log")
        iter_num = None
        
        if os.path.exists(logfile_path):
            with open(logfile_path, 'r') as f:
                for line in f:
                    if "[train iter]:" in line:
                        try:
                            # Extract iteration number (e.g., from "156/500")
                            iter_str = line.split("[train iter]:")[1].strip().split('/')[0]
                            iter_num = int(iter_str)
                            break
                        except (IndexError, ValueError):
                            continue
        
        if iter_num is None:
            print(f"Warning: Could not find iteration number for task {task_id}")
            continue
            
        # Check output.json for success
        output_file = os.path.join(task_path, "output.json")
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    successes_by_iter[iter_num] = 1 if data.get('reward', False) else 0
            except (json.JSONDecodeError, FileNotFoundError):
                continue

    print(f"Completed processing {total_tasks} tasks")
    print("Calculating rolling statistics...")

    # Sort successes by iteration number
    sorted_successes = dict(sorted(successes_by_iter.items()))
    
    # Calculate rolling successes and total attempts
    rolling_successes = []
    total_attempts = []
    cumulative_success = 0
    cumulative_attempts = 0
    
    for iter_num in sorted_successes.keys():
        cumulative_success += sorted_successes[iter_num]
        cumulative_attempts += 1
        rolling_successes.append(cumulative_success)
        total_attempts.append(cumulative_attempts)

    return {
        'successes_by_iter': sorted_successes,
        'rolling_successes': rolling_successes,
        'total_attempts': total_attempts,
        'iterations': list(sorted_successes.keys()),
        'total_tasks': total_tasks
    }
