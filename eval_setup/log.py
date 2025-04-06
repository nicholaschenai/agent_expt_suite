import random
import sys
import logging

from pprint import pp
from typing import Dict, Any, List
from rich import print as rprint

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from cognitive_base.utils import color_mapping

logger = logging.getLogger("logger")


class VerboseHandler(BaseCallbackHandler):
    """
    handler for langchain callbacks
    """
    def __init__(self, verbose=False):
        self.color_list = color_mapping
        self.color = None
        self.ENDC = '\033[0m'
        self.verbose = verbose
        self.task_id = None

    # sorta deprecated
    def print_prompts(self, prompts):
        self.set_rand_color()
        print('\n===start of prompt===\n')
        print('\n===msg delimiter===\n'.join([prompt.content for prompt in prompts]))
        print('\n===end of prompt===\n')
        self.reset_color()

    def print_response(self, response):
        """
        Prints the response (langchain) from a language model (LLM) in a formatted manner.

        Args:
            response (Response): The response object containing generations, llm_output, and run information.

        The method performs the following steps:
        1. Sets a random color for the output.
        2. Prints the start of the LLM response.
        3. Iterates through each generation in the response and prints its text, generation info, type, and other relevant details.
        4. Prints the LLM output.
        5. Prints the run information.
        6. Resets the color after printing the response.
        """
        self.set_rand_color()
        print('===start of LLM response===\n')
        for lst in response.generations:
            for generation in lst:
                print('generation text\n')
                print(generation.text)
                print('generation info\n')
                pp(generation.generation_info)
                print('\n')
                print('generation type\n')
                print(generation.type)
                print('other generation stuff\n')
                d = generation.dict()
                del d['message']['content']
                # pp({k: v for k, v in d.items() if k not in ['text', 'generation_info', 'type']})
                rprint({k: v for k, v in d.items() if k not in ['text', 'generation_info', 'type']})
                print('\n')
        print('llm_output\n')
        pp(response.llm_output)
        print('\n')
        print('run\n')
        pp(response.run)
        print('\n')
        print('===end of LLM response===\n')
        self.reset_color()

    def set_rand_color(self):
        _, color = random.choice(self.color_list)
        if color == self.color:
            return self.set_rand_color()
        self.color = color
        sys.stdout.write(self.color)

    def reset_color(self):
        sys.stdout.write(self.ENDC)

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        # if self.verbose:
        #     self.print_prompts(prompts)
        for prompt in prompts:
            logger.info(f'[Task id] {self.task_id} [prompt] {prompt}')

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        if self.verbose:
            self.print_response(response)
        for lst in response.generations:
            for generation in lst:
                logger.info(f'[Task id] {self.task_id} [generation text] {generation.text}')

    def on_chat_model_start(
            self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""
        print('\n===start of messages===\n')
        for lst in messages:
            # if self.verbose:
            #     self.print_prompts(lst)
            for msg in lst:
                logger.info(f'[Task id] {self.task_id} [prompt] {msg.content}')
        print('\n===end of messages===\n')
