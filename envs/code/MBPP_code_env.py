from .base_code_env import BaseCodeEnv

from .executors.py_executor import PyExecutor


class MbppCodeEnv(BaseCodeEnv):
    """
    A class to represent a coding environment that manages the execution of code snippets
    against a set of tests for specific datasets (e.g., MBPP, APPS).

    Attributes:
        exe (PyExecutor): An executor to run Python code snippets.
        timeout (int): The maximum time allowed for code execution.
        tests (list): A list of test cases for the current task.
        test_prefix (str): Prefix to add before each test case, e.g., setup code or imports.
        mbpp_plus_data (tuple): Data specific to the MBPP+ dataset, if applicable.
        use_public_tests (bool): Flag to use public tests instead of private ones.
        public_tests (list): A list of public test cases.
        dataset_name (str): The name of the dataset being used.
        max_tests (int): The maximum number of tests to display in feedback.
        max_chars (int): The maximum number of characters to display per test in feedback.
    """
    def __init__(
            self,
            timeout=10,
            max_display_tests=None,
            max_display_chars=None,
            use_public_tests=False,
            dataset_name="MBPP",
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
        self.exe = PyExecutor(max_display_tests=max_display_tests, max_display_chars=max_display_chars)
        self.mbpp_plus_data = ()

    def _reset(self, task):
        # this is for mbpp, where test setup code might be required
        if 'test_setup_code' in task:
            self.test_prefix = task['test_setup_code'] + '\n'
        elif 'test_imports' in task:
            self.test_prefix = '\n'.join(task['test_imports']) + '\n'
        else:
            self.test_prefix = ''

        self.private_tests = task['test_list']
        if 'mbpp_plus_problem' in task:
            self.mbpp_plus_data = (task['mbpp_plus_problem'], task['mbpp_plus_output'])
        else:
            self.mbpp_plus_data = ()

    def _step(self, full_code, use_public_tests=False):
        """
        Executes the given code against the test cases and generates feedback.

        Parameters:
            full_code (str): The complete code snippet to be executed.
            use_public_tests (bool): Flag indicating whether to use public tests.
        """
        exe_out = self.exe.execute(
            func=full_code + '\n' + self.test_prefix,
            tests=self.public_tests if use_public_tests else self.private_tests,
            timeout=self.timeout,
            mbpp_plus_data=() if use_public_tests else self.mbpp_plus_data,
            use_public_tests=use_public_tests
        )
        reward, obs, individual_results = exe_out

        return obs, reward, None, {'individual_results': individual_results}
    

