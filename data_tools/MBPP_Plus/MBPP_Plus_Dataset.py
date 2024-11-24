"""
Dataset to be used for MBPP
"""
import torch
import json
import copy

from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS


class MBPPPlusDataset(torch.utils.data.Dataset):
    """
    A custom Dataset class for the MBPP Plus dataset.

    Attributes:
        problems (dict): Dictionary to store the problems.
        dataset_hash (str): Hash of the dataset.
        expected_output (dict): Dictionary to store the expected outputs.
        raw_ids (list): List of raw problem IDs.
        dataset_name (str): Name of the dataset.
        eval_later (bool): Flag to indicate if evaluation is to be done at the end in an entire batch (if False, evaluation is done per problem).
        use_public_tests (bool): Flag to indicate if public tests are used.
    """

    def __init__(self, dataset_name="MBPP_Plus", eval_later=False, use_public_tests=False):
        """
        Initializes the MBPPPlusDataset.

        Args:
            dataset_name (str): Name of the dataset. Default is "MBPP_Plus".
            eval_later (bool): Flag to indicate if evaluation is to be done  the end in an entire batch (if False, evaluation is done per problem). Default is False.
            use_public_tests (bool): Flag to indicate if public tests are to be used. Default is False.
        """
        self.problems = {}
        self.dataset_hash = None
        self.expected_output = {}
        self.raw_ids = []
        self.dataset_name = dataset_name
        self.eval_later = eval_later
        self.use_public_tests = use_public_tests

        noextreme = 'noextreme' in dataset_name
        if noextreme:
            print("noextreme")
        self.problems = get_mbpp_plus(noextreme=noextreme)
        if not (self.use_public_tests and self.eval_later):
            dataset_hash = get_mbpp_plus_hash(noextreme=noextreme)
            self.expected_output = get_groundtruth(
                self.problems,
                dataset_hash,
                MBPP_OUTPUT_NOT_NONE_TASKS,
            )
        self.raw_ids = list(self.problems.keys())
        # so reference by idx is deterministic
        self.raw_ids.sort()

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of problems in the dataset.
        """
        return len(self.raw_ids)

    def __getitem__(self, idx):
        """
        Returns a data point from the dataset at the given index.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            dict: Data point at the given index.
        """
        return self.get_mbpp_plus_datapoint(idx)

    def get_mbpp_plus_datapoint(self, idx):
        """
        Retrieves a data point from the dataset.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            dict: Data point at the given index.
        """
        raw_id = self.raw_ids[idx]
        test_datapoint = {}
        raw_datapoint = self.problems[raw_id]

        if not (self.use_public_tests and self.eval_later):
            raw_ans = self.expected_output[raw_id]
            test_datapoint['mbpp_plus_problem'] = copy.deepcopy(raw_datapoint)
            test_datapoint['mbpp_plus_output'] = copy.deepcopy(raw_ans)

        test_datapoint['task_id'] = raw_datapoint['task_id']

        intro_str = ("Write a Python function that satisfies the description below."
                     " Your code must strictly follow the function name and typings of the input and output "
                     "specified in the assert statement below, and pass the assertion when executed.")
        test_datapoint['task'] = raw_datapoint['prompt']
        test_datapoint['task_prompt'] = intro_str + '\n' + raw_datapoint['prompt']
        test_datapoint['gt_fn_name'] = raw_datapoint['entry_point']
        test_datapoint['code'] = raw_datapoint['canonical_solution']

        test_datapoint["test_list"] = [''] * (len(raw_datapoint['base_input']) + len(raw_datapoint['plus_input']))

        if self.use_public_tests:
            assert_statement = ''
            for line in raw_datapoint['prompt'].split('\n'):
                if 'assert' in line and raw_datapoint['entry_point'] in line:
                    assert_statement = line
                    break
            if assert_statement:
                test_datapoint["public_test_list"] = [assert_statement]
            else:
                raise ValueError(f"no assert in id {raw_datapoint['task_id']}. prompt:\n{raw_datapoint['prompt']}")
        return test_datapoint
