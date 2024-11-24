from evalplus.data import write_jsonl

from cognitive_base.utils import load_json
from cognitive_base.utils.log import construct_task_folder

def postprocess_mbpp_plus(dataloader, dataset, result_dir):
    """
    Post-processes the MBPP Plus results for eval by consolidating into a JSONL file.

    Args:
        dataloader (DataLoader): The data loader providing batches of data.
        dataset (Dataset): The dataset being processed.
        result_dir (str): The directory where the results are stored.

    Returns:
        None

    This function iterates over the batches provided by the dataloader, extracts the task ID and the corresponding
    solution from the output JSON files, and consolidates them into a single JSONL file named 'samples.jsonl' in the
    specified result directory.
    """
    # consolidate into samples.jsonl to evaluate
    samples = []
    print('consolidating MBPP Plus samples into jsonl')
    for i, batch in enumerate(dataloader):
        full_task = list(batch)[0]
        task_id = str(full_task['task_id'])
        task_folder = construct_task_folder(result_dir, 'test', task_id)
        output_d = load_json(f"{task_folder}/output.json")
        samples.append({'task_id': task_id, 'solution': output_d.get('full_code', '')})
    write_jsonl(f"{result_dir}/samples.jsonl", samples)
    