"""
Heavily adapted from reflexion

Modifications:
- globals() replaced as it  might cause override of default functions like sum
- more importantly, a note on security: typical vulnerabilities are
    - enabling globals which can expose api keys
    - modules (eg pickle) that enable loading of files (these files can potentially be malicious)
    - os module which can access filesystem / reverse shell
- so we modify the global variable to exec with a safer version
"""
import ast
import astunparse

from typing import List
from RestrictedPython import safe_builtins, utility_builtins
from evalplus.evaluate import check_correctness

from .executor_utils import function_with_timeout
from .executor_types import ExecuteResult, Executor
from .humaneval_execution import swallow_io, create_tempdir

from cognitive_base.utils import load_json
from cognitive_base.utils.code_parse import whitelist_modules_path

other_whitelisted_builtins = {
    "min": min,
    "max": max,
    "map": map,
    "sum": sum,
    "type": type,
    "next": next,
    "filter": filter,
    "all": all,
    "enumerate": enumerate,
    "dict": dict,
    "any": any,
    "bin": bin,
    "range": range,
    "list": list,
    "tuple": tuple,
    "reversed": reversed,
    "iter": iter,
    "object": object,
    "print": print,
    }

try:
    whitelist_modules = load_json(whitelist_modules_path)
    _SAFE_MODULES = frozenset(whitelist_modules)


    def _safe_import(name, *args, **kwargs):
        print(f'module name {name}\n')
        print(f'__module__ name {name}\n')
        if name not in _SAFE_MODULES:
            raise Exception(f"module not in whitelist: {name!r}")
        return __import__(name, *args, **kwargs)


    my_globals = {
        "__builtins__": {
            **safe_builtins,
            **utility_builtins,
            **other_whitelisted_builtins,
            "__import__": _safe_import,
        },
        "__name__": __name__,
    }
    print('whitelisted modules loaded')
except:
    print('no whitelisted modules, defaulting to locals() for global variables during code execution')
    my_globals = locals()


def get_globals():
    # in case function names override builtins eg 'sum'
    return my_globals.copy()


def get_test(tests, mbpp_plus_data, i):
    if mbpp_plus_data:
        if i < len(mbpp_plus_data[0]['base_input']):
            fn_input = mbpp_plus_data[0]['base_input'][i]
            fn_output = mbpp_plus_data[1]['base'][i]
        else:
            j = i - len(mbpp_plus_data[0]['base_input'])
            fn_input = mbpp_plus_data[0]['plus_input'][j]
            fn_output = mbpp_plus_data[1]['plus'][j]
        return f"assert {mbpp_plus_data[0]['entry_point']}({repr(fn_input)[1:-1]})=={repr(fn_output)}"
    else:
        return tests[i]


class PyExecutor(Executor):
    def __init__(self, max_display_tests=None, max_display_chars=None):
        # truncate incase output is long.
        # since we usually care about passing all test cases, can use this to early stop if too many mistakes
        self.max_tests = max_display_tests
        self.max_chars = max_display_chars

    def execute(self, func: str, tests: List[str], timeout: int = 5, mbpp_plus_data: tuple = (),
                use_public_tests: bool = False) -> ExecuteResult:
        # Combine function code and assert statement
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

        # Run the tests and collect the results
        # success_tests = []
        # failed_tests = []
        success_test_idxs = []
        failed_test_idxs = []
        failed_test_outputs = []
        is_passing = True
        num_tests = len(func_test_list)
        if mbpp_plus_data:
            ret = check_correctness(
                "mbpp",
                0,
                mbpp_plus_data[0],
                func,
                mbpp_plus_data[1],
                False,
                min_time_limit=1,
                gt_time_limit_factor=4
                )
            # print('done')
            res = ret['base'][1] + ret['plus'][1]

        state = []
        for i in range(num_tests):
            # print(i)
            # print(res[i])
            with create_tempdir():
                with swallow_io():
                    try:
                        if mbpp_plus_data:
                            assert res[i] == 1
                        else:
                            function_with_timeout(exec, (func_test_list[i], get_globals()), timeout)
                        # success_tests += [tests[i]]
                        if self.max_tests and len(success_test_idxs) < self.max_tests:
                            success_test_idxs.append(i)
                        state.append(True)
                    except Exception:
                        test = get_test(tests, mbpp_plus_data, i)
                        output = get_output(func, test, timeout=timeout)
                        # output = ''
                        # failed_tests += [f"{tests[i]} # output: {output}"]
                        failed_test_idxs.append(i)
                        failed_test_outputs.append(output)
                        is_passing = False
                        state.append(False)
                        if self.max_tests and len(failed_test_idxs) >= self.max_tests:
                            break

        # for test in tests:
        #     if test in success_tests:
        #         state += [True]
        #     else:
        #         state += [False]

        state = tuple(state)

        feedback = 'Note: Tests are automatically generated and can be wrong.\n\n' if use_public_tests else ''
        feedback += "Tests passed:"
        # for test in success_tests:
        #     feedback += f"\n{test}"
        for idx in success_test_idxs:
            test_str = get_test(tests, mbpp_plus_data, idx)
            if self.max_chars and len(test_str) > self.max_chars:
                test_str = test_str[:self.max_chars] + '...'
            feedback += f"\n{test_str}"
        if not success_test_idxs:
            feedback += f"\nNone"
        feedback += "\n\nTests failed:"
        # for test in failed_tests:
        #     feedback += f"\n{test}"
        for i, test_idx in enumerate(failed_test_idxs):
            test_str = get_test(tests, mbpp_plus_data, test_idx)
            output_str = str(failed_test_outputs[i])
            if self.max_chars:
                if len(test_str) > self.max_chars:
                    test_str = test_str[:self.max_chars] + '...'
                if len(output_str) > self.max_chars:
                    output_str = output_str[:self.max_chars] + '...'
            feedback += f"\n{f'{test_str} # output: {output_str}'}"
        if not failed_test_idxs:
            feedback += "\nNone"
        return ExecuteResult(is_passing, feedback, state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        code = f"""{func}

{test}

check({name})
    """
        print('temporarily disabled')

        # try:
        #     function_with_timeout(exec, (code, get_globals()), timeout)
        #     return True
        # except Exception:
        #     return False


def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left  # type: ignore
    except:
        call_str = ast_parsed.body[0].test  # type: ignore

    return astunparse.unparse(call_str).strip()


def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        # this time, we need globals to persist as the execution is in 2 stages
        e_globals = get_globals()
        exec(f"from typing import *\n{func}", e_globals)
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, e_globals), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    # Test the function
    func = "def add(a, b):\n    while True:\n        x = 1\n    return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(1, 2) == 4"]
    print(PyExecutor().execute(func, tests, timeout=1))
