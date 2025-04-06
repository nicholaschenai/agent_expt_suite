"""
Functions to evaluate MBPP Plus results
"""

def get_overall_status(statuses):
    """
    Function to determine overall status (both base and plus)
    """
    return 'pass' if all(status == 'pass' for status in statuses) else 'fail'


def unequal_status(d):
    """
    Function to check and print task_ids where base_status is not equal to plus_status
    """
    count = 0
    score = 0
    for k, v in d['eval'].items():
        if v[0]['base_status'] != v[0]['plus_status']:
            print(f"task_id: {k}")
            count += 1
        if get_overall_status([v[0]['base_status'], v[0]['plus_status']]) == 'pass':
            score += 1
    print(f'count: {count}')
    print(f'score: {score}')


def fail_status(d):
    """
    Function to check and print task_ids where overall status is fail
    """
    count = 0
    for task_id, v in d['eval'].items():
        if get_overall_status([v[0]['base_status'], v[0]['plus_status']]) == 'fail':
            print(f"{count}. task_id: {task_id}")
            count += 1
    print(f'count: {count}')


def compare_dict_statuses(dict1, dict2, dict1_name="dict1", dict2_name="dict2"):
    """Compare overall statuses between two result dictionaries and print differences.

    Args:
        dict1: First result dictionary
        dict2: Second result dictionary 
        dict1_name: Display name for first dictionary (default: "dict1")
        dict2_name: Display name for second dictionary (default: "dict2")
        
    Returns:
        dict: Dictionary mapping (status1, status2) tuples to lists of task IDs
    """
    # Initialize result dictionary to store status combinations
    status_groups = {}
    
    for k, v in dict1['eval'].items():
        status1 = get_overall_status([v[0]['base_status'], v[0]['plus_status']])
        status2 = get_overall_status([dict2['eval'][k][0]['base_status'], dict2['eval'][k][0]['plus_status']])
        
        # Add to status groups dictionary
        status_pair = (status1, status2)
        if status_pair not in status_groups:
            status_groups[status_pair] = []
        status_groups[status_pair].append(k)

    # Print differences grouped by status
    print(f"\nTask IDs where overall status differs between {dict1_name} and {dict2_name}, grouped by status:")
    for (s1, s2), task_ids in status_groups.items():
        if s1 != s2:
            print(f"\n{dict1_name}: {s1}, {dict2_name}: {s2}")
            print(f"Count: {len(task_ids)}")
            print(f"Task IDs: {task_ids}")
    
    return status_groups
