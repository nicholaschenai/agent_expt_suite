from collections import defaultdict
import os
import json

def get_error_type_stats(folder: str) -> dict:
    """
    Analyzes error types in APPS evaluation results.
    
    Args:
        folder: Path to the base results folder
        
    Returns:
        Dict containing error type counts and descriptions
    """
    error_counts = defaultdict(int)
    
    # Iterate through task folders
    for task_id in os.listdir(folder):
        task_path = os.path.join(folder, task_id)
        if not os.path.isdir(task_path):
            continue
            
        output_file = os.path.join(task_path, "output.json")
        if not os.path.exists(output_file):
            continue
            
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                full_state = data.get('state')
                # TODO: dont just take the first state as it can be mixed
                state = full_state[0]
                
                if state is True:
                    error_counts['correct'] += 1
                elif state == -1:
                    error_counts['runtime_error_tle'] += 1
                elif state == -2:
                    error_counts['compile_error'] += 1
                else:
                    error_counts['wrong_answer'] += 1
        except (json.JSONDecodeError, FileNotFoundError):
            continue

    return {
        'counts': dict(error_counts),
        'total': sum(os.listdir(folder))
    }
