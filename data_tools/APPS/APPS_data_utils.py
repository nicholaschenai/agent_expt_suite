import copy
import logging
import json

from ...envs.code.executors.utils_evaluate import safe_eval_answer_from_agent


logger = logging.getLogger("logger")


apps_task_template = """
The output code needs to {question_guide}
{starter_str}
"""


def apps_preprocess_test(full_task):
    example = copy.deepcopy(full_task)
    input_output = example['input_output']
    try:
        if type(input_output) == str:
            tests = json.loads(input_output)
        else:
            tests = input_output
        example['gt_fn_name'] = tests.get('fn_name', '')
    except:
        logger.info(f"Failed to get unit tests for problem {example['problem_id']} with {input_output}")
    example['task'] = example['question']
    example['task_id'] = example['problem_id']
    starter_code = example['starter_code']
    if starter_code:
        question_guide = 'use the provided function signature'
        starter_str = f"\nThe final python function should begin with: \n```python\n{starter_code}\n```"
    else:
        question_guide = 'read from and write to standard IO'
        starter_str = ''
    apps_task_str = apps_task_template.format(question_guide=question_guide, starter_str=starter_str)
    example['task_prompt'] = example['task'] + '\n' + apps_task_str
    return example


def filter_long_solns(solns, task, max_len=36000):
    """
    Filter solutions to select one that, combined with the task description, does not exceed a maximum length.

    Args:
        solns (list): A list of candidate solution strings.
        task (str): The task description.
        max_len (int, optional): max len of task description plus solution. Defaults to 36000.

    Returns:
        str: The selected solution or an empty string if none are suitable.
    """
    soln = ''
    for soln_idx, candidate_soln in enumerate(solns):
        if len(task + candidate_soln) < max_len:
            soln = candidate_soln
            print(f'using soln:\n {soln[:1000]}...\n')
            break
    return soln, soln_idx


def apps_preprocess_train(full_task, code='', soln_idx=None):
    example = apps_preprocess_test(full_task)
    solns, qn = json.loads(example['solutions']), example['question']
    if code:
        example['code'] = code
    elif soln_idx is not None:
        example['code'] = solns[soln_idx]
    else: 
        example['code'] = filter_long_solns(solns, qn)[0]
    return example


def check_missing_or_long(example, max_len):
    p_id, task, solns = example['problem_id'], example['question'], json.loads(example['solutions'])
    if not task or not solns:
        print(f"data point {p_id} not loaded as it is missing qn or soln\n")
        return True, '', '', '', ''
    soln, soln_idx = filter_long_solns(solns, task, max_len)
    if not soln:
        print(f"data point {p_id} not loaded as it exceeds max len\n")
        return True, '', '', '', ''
    return False, soln, p_id, task, soln_idx


def apps_filter_fn(example, max_len=36000):
    """
    Filter out data points that do not have valid solutions, test cases, or exceed the maximum length.
    """
    discard, soln, p_id, task, soln_idx = check_missing_or_long(example, max_len)
    if discard:
        return False, ''
    example['gpt_codes'] = [soln]
    example['soln_idx'] = soln_idx
    example = safe_eval_answer_from_agent(example, debug=True, return_output=True)
    try:
        if type(example['input_output']) == str:
            tests = json.loads(example['input_output'])
        else:
            tests = example['input_output']
        if (not tests.get("inputs", [])) and (not tests.get("outputs", [])):
            print(f'missing tests for id {p_id}\n')
            return False, ''
    except:
        print(f"Failed to get unit tests for problem {p_id} with {example['input_output']}\n")
        return False, ''
    if not example['gpt_pass_flags']:
        print(f"execution error {p_id}\n")
        return False, ''
    if not example['gpt_pass_flags'][0]:
        print(f"solution gives wrong ans {p_id}\n")
        return False, ''
    return True, p_id
