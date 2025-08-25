"""
I/O utilities for experimental evaluation
"""
import os
import shutil
import csv
import logging
import json

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def save_status_difference_logs(
    wrong_to_right: List[str],
    right_to_wrong: List[str],
    status_cols: List[str],
    exp1_name: str,
    exp2_name: str,
    model_name: str,
    output_base_dir: str
) -> None:
    """
    Save log files for tasks that have different statuses between two experiments.
    
    Args:
        wrong_to_right: List of task IDs where exp1 failed but exp2 passed
        right_to_wrong: List of task IDs where exp1 passed but exp2 failed
        status_cols: List of status column names from the CSV
        exp1_name: Name of first experiment
        exp2_name: Name of second experiment
        model_name: Name of the model
        output_base_dir: Base directory to save outputs
    """
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
            # TODO: shouldnt be final results folder, make it flexible
            src_log = os.path.join("final_results", f"{exp_name}_{model_name}", 
                                 "test_outputs", safe_task_id, "logfile.log")
            dst_log = os.path.join(task_dir, f"{exp_name}.log")
            
            if os.path.exists(src_log):
                shutil.copy2(src_log, dst_log)
            else:
                print(f"Warning: Log file not found for {task_id} in {exp_name}")


def save_status_differences_to_csv(
    status_groups: Dict[Tuple[str, str], List[str]],
    filename: str,
    exp1_status_col: str,
    exp2_status_col: str
) -> None:
    """Save status differences to CSV file.

    Args:
        status_groups: Dictionary mapping status tuples to problem IDs
        filename: Output CSV filename
        exp1_status_col: Column name for first experiment status
        exp2_status_col: Column name for second experiment status
    """
    logger.info(f"Saving status differences to {filename}")
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['problem_id', exp1_status_col, exp2_status_col]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for (s1, s2), task_ids in status_groups.items():
            if s1 != s2:
                for task_id in task_ids:
                    writer.writerow({'problem_id': task_id, exp1_status_col: s1, exp2_status_col: s2})

def load_evaluation_result(experiment_path: Path) -> Optional[Dict[str, Any]]:
    """Load evaluation results for a single experiment of MBPP plus.
    
    Args:
        experiment_path: Path to experiment directory
        
    Returns:
        Evaluation results dictionary or None if not found
    """
    eval_results_file = experiment_path / 'samples_eval_results.json'
    if not eval_results_file.exists():
        raise ValueError(f"Evaluation results not found for experiment: {experiment_path}")
    
    with open(eval_results_file, 'r') as f:
        return json.load(f)
