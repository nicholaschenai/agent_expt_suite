import copy
from pprint import pp
from .executors.utils_evaluate import safe_eval_answer_from_agent
from .base_code_env import BaseCodeEnv


class AppsCodeEnv(BaseCodeEnv):
    """
    A class to represent a coding environment that manages the execution of code snippets
    against a set of tests for specific datasets (e.g., MBPP, APPS).

    Attributes:
        timeout (int): The maximum time allowed for code execution.
        tests (list): A list of test cases for the current task.
        test_prefix (str): Prefix to add before each test case, e.g., setup code or imports.
        use_public_tests (bool): Flag to use public tests instead of private ones.
        public_tests (list): A list of public test cases.
        APPS_datapoint (dict): Data specific to the APPS dataset, if applicable.
        dataset_name (str): The name of the dataset being used.
        max_tests (int): The maximum number of tests to display in feedback.
        max_chars (int): The maximum number of characters to display per test in feedback.
        enable_input_hints (bool): Flag to enable input handling hints.
    """
    def __init__(
        self,
        timeout=10,
        max_display_tests=None,
        max_display_chars=None,
        use_public_tests=False,
        dataset_name="APPS",
        enable_input_hints=False,
        **kwargs
    ):
        """
        Initializes the CodeEnv with the given parameters.

        Parameters:
            timeout (int): The maximum time allowed for code execution.
            max_display_tests (int): The maximum number of tests to display in feedback.
            max_display_chars (int): The maximum number of characters to display per test in feedback.
            use_public_tests (bool): Flag to use public tests instead of private ones.
            do_train (bool): Flag indicating if the environment is in training mode.
            do_test (bool): Flag indicating if the environment is in testing mode.
            dataset_name (str): The name of the dataset being used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            timeout=timeout,
            max_display_tests=max_display_tests,
            max_display_chars=max_display_chars,
            use_public_tests=use_public_tests,
            dataset_name=dataset_name,
            **kwargs,
        )
        self.generic_code_env = True
        self.APPS_datapoint = {}
        self.enable_input_hints = enable_input_hints

    def _reset(self, task):
        self.APPS_datapoint = copy.deepcopy(task)

    def construct_env_feedback(self, outcomes, all_outputs, use_public_tests):
        """
        Constructs feedback for the executed code based on the outcomes of the test cases.

        Parameters:
            outcomes (list): A list of int values indicating the outcomes of the test cases.
            all_outputs (list): A list of tuples containing the execution output, expected output,
                                and test input for each test case.
            use_public_tests (bool): Flag indicating whether public tests were used.

        Returns:
            str: A string containing the constructed feedback.
        """
        feedback = 'Note: Tests are automatically generated and can be wrong.\n\n' if use_public_tests else ''
        feedback += 'Note: Inputs/outputs here are automatically extracted/truncated so formatting may be a bit off.\n'
        feedback += "Tests passed:\n"
        fail_str = "\n\nTests failed:"
        # print(f'all outputs {all_outputs}')

        success_display, fail_display = 0, 0
        for outcome, all_output in zip(outcomes, all_outputs):
            exe_output, expected_output, test_input = all_output
            exe_output, expected_output, test_input = str(exe_output), str(expected_output), str(test_input)
            if outcome == True:
                # outcome can be -1 or -2 if error
                if self.max_tests and success_display < self.max_tests:
                    feedback += (
                        f"\n Input: {test_input[:self.max_chars // 2]} "
                        f"Output: {expected_output[:self.max_chars // 2]}"
                        )
                    success_display += 1
            else:
                if self.max_tests and fail_display < self.max_tests:
                    fail_str += (
                        f"\n Input: {test_input[:self.max_chars // 3]} "
                        f"Expected output: {expected_output[:self.max_chars // 3]} "
                        f"# Execution output: {exe_output[:self.max_chars // 3]}"
                        )
                    if outcome == -1:
                        fail_str += ' # Runtime error or time limit exceeded error'
                    if outcome == -2:
                        fail_str += ' # Compile Error'
                    # Add input handling hint if enabled and output is blank/empty
                    if exe_output == '[]' and self.enable_input_hints:
                        fail_str += input_hint_str
                    fail_display += 1
            if self.max_tests and (fail_display >= self.max_tests) and (success_display >= self.max_tests):
                break

        if not success_display:
            feedback += "\nNone"
        feedback += fail_str
        if not fail_display:
            feedback += "\nNone"

        return feedback

    def _step(self, full_code, use_public_tests=False):
        """
        Executes the given code against the test cases and generates feedback.

        Parameters:
            full_code (str): The complete code snippet to be executed.
            use_public_tests (bool): Flag indicating whether to use public tests.

        Returns:
            ExecuteResult: An object containing the execution result, feedback, and state.
        """
        # env_out = ExecuteResult(False, '\n Error during execution\n', (False,))
        obs, reward, _, individual_results = '\n Error during execution\n', False, False, (False,)

        example = copy.deepcopy(self.APPS_datapoint)
        example['gpt_codes'] = [full_code]
        example = safe_eval_answer_from_agent(example, return_output=True)
        if example.get('details', ''):
            outcomes, all_outputs = example['details'][0]
            obs = self.construct_env_feedback(outcomes, all_outputs, use_public_tests)
            # individual_results = tuple([res == True for res in outcomes])  # can be -1 or -2 to indicate errors
            individual_results = outcomes
            # env_out = ExecuteResult(example['gpt_pass_flags'][0], feedback, state)
            reward = example['gpt_pass_flags'][0]

        return obs, reward, None, {'individual_results': individual_results}

input_hint_str = """
No output detected. You might want to check the reading from / writing to standard IO.
A common mistake is to put the IO inside a function, but the function is not called.
"""
