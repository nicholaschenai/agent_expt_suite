import json
import os

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
