import json
import os
import shutil
import logging
import glob

from pathlib import Path

from .eval_manager import EvalManager

from ..envs import env_globals
from ..envs.env_registry import env_registry

from ..data_tools.dataset_registry import dataset_registry
from ..data_tools.data_pipeline_factory import data_pipeline_factory

from cognitive_base.utils import custom_breakpoint, f_mkdir, load_json, dump_json

logger = logging.getLogger("logger")


# Initializing stuff
def initialize_directories(args):
    """
    Initializes directories for training and checkpoint management based on the provided arguments.
    Args:
        args: An object containing the following attributes:
            do_train (bool): Flag indicating whether to create training directories.
            result_dir (str): The base directory for storing results.
            ckpt_dir (str): The directory for storing checkpoints. If not provided, it will be set based on other arguments.
            standalone_ckpt (bool): Flag indicating whether the checkpoint directory is standalone.
            ckpt_name (str): The name of the checkpoint directory.
            clone_ckpt (str): The directory from which to clone an existing checkpoint.
    Logs:
        Logs the checkpoint directory path and any warnings or information related to cloning checkpoints.
    Raises:
        Custom breakpoint calls are made for debugging purposes if certain conditions are met.
    """
    if args.do_train:
        Path(f"{args.result_dir}/saved_train_ckpt").mkdir(parents=True, exist_ok=True)

    if not args.ckpt_dir:
        args.ckpt_dir = ((args.result_dir + '/') if not args.standalone_ckpt else '') + args.ckpt_name
    logger.info(f'[ckpt dir] {args.ckpt_dir}')

    # clone checkpoint to continue from previous state
    if args.clone_ckpt:
        if os.path.isdir(args.ckpt_dir):
            logger.warning("ckpt_dir exists, skip cloning checkpoint")
            custom_breakpoint()
            # do not clone checkpoint if ckpt exists, incase we resume training we dont wna override
        elif os.path.isdir(args.clone_ckpt):
            logger.info(f'clone checkpoint from {args.clone_ckpt}')
            shutil.copytree(args.clone_ckpt, args.ckpt_dir, dirs_exist_ok=True)
        else:
            logger.warning(f'clone ckpt failed. {args.clone_ckpt} is not a directory')
            custom_breakpoint()


def initialize_environment(args, env_registry_updates=None):
    """
    Initializes the environment with the given arguments and optional registry updates.
    Supports separate train and test environments.

    Args:
        args (Namespace): A namespace object containing the arguments for initializing the environment.
        env_registry_updates (dict, optional): A dictionary containing updates to the environment registry.

    Returns:
        None
    """
    kwargs = vars(args)
    args.generic_code_env = False
    if args.do_train or args.do_test or args.do_val:
        if env_registry_updates:
            env_registry.update(env_registry_updates)
        env_globals.init_code_env(**kwargs)
        # Set generic_code_env based on current task_env
        args.generic_code_env = env_globals.task_env.generic_code_env


def save_args(args):
    dump_json(vars(args), f"{args.result_dir}/args.json", indent=4)


def attach_env_to_agent(args, actor):
    """
    Attaches the appropriate environment to the agent based on whether we're training or testing.
    
    Args:
        args: Arguments containing do_train and do_test flags
        actor: The agent to attach the environment to
    """
    # TODO: current flow only accepts either train or test script
    # possible to have both at one go but need to sort out the generic_code_env
    # that is used in the agent init 
    # (need to re-init agent stuff if train n test envs are different in that val)
    if args.do_train:
        actor.env_interface = env_globals.train_env
    elif args.do_test or args.do_val:
        actor.env_interface = env_globals.test_env


# train related
def load_train_data(args, actor, dataset_registry_updates=None):
    """
    Evaluates the performance of an agent using the provided arguments and actor.
    Args:
        args (Namespace): A namespace object containing various arguments and configurations.
            - do_test (bool): Flag to indicate whether to perform testing.
            - test_dataset_name (str, optional): Name of the test dataset to use.
            - dataset_name (str): Name of the dataset to use (legacy compatibility)
        actor (object): The agent or model to be evaluated.
        dataset_registry_updates (dict, optional): Updates to be applied to the dataset registry.
    Returns:
        None
    """
    if args.do_train and args.load_train:
        if args.train_dataset_name:
            args.dataset_name = args.train_dataset_name
        kwargs = vars(args)
        
        if dataset_registry_updates:
            print("Before dataset_registry update:", json.dumps(dataset_registry, indent=4))
            dataset_registry.update(dataset_registry_updates)
            print("After update:", json.dumps(dataset_registry, indent=4))
        data_pipeline = data_pipeline_factory(train=True, **kwargs)
        data_pipeline.attach_to_agent(actor)


def prepare_train(args, actor):
    """
    Prepare training by creating necessary directories and optionally resuming from a checkpoint.

    Args:
        args: Command line arguments or any arguments object with necessary attributes.
        actor: The actor or model to be trained, which will be updated if resuming training.
    """
    f_mkdir(f"{args.result_dir}/train_outputs/")
    if args.resume:
        data = load_json(f"{args.result_dir}/train_ckpt_info.json")
        if data:
            for attr, value in data.items():
                if value:
                    setattr(actor, attr, value)
        elif 'voyager' in args.agent_type:
            # legacy compatibility
            actor.train_iter = len(glob.glob(f"{args.result_dir}/train_outputs/*.json"))
        print(f'resume training from iteration {actor.train_iter}')


# def validate_agent(args, actor, dataset_registry_updates=None):
#     """
#     Validates the agent's performance using the validation split of the training data.
#     Args:
#         args (Namespace): A namespace object containing various arguments and configurations.
#         actor (object): The agent or model to be validated.
#         dataset_registry_updates (dict, optional): Updates to be applied to the dataset registry.
#     Returns:
#         float: The validation accuracy
#     """
#     if args.train_dataset_name:
#         args.dataset_name = args.train_dataset_name
#     kwargs = vars(args)
#     kwargs['validation'] = True  # Set validation mode
    
#     if dataset_registry_updates:
#         dataset_registry.update(dataset_registry_updates)
#     data_pipeline = data_pipeline_factory(train=True, **kwargs)

#     eval_manager = EvalManager(args, actor, data_pipeline)
#     eval_manager.test_loop()
    
#     # Calculate validation accuracy
#     result_dict = load_json(f"{args.result_dir}/result_dict.json")
#     if result_dict:
#         result_val = result_dict.values()
#         val_accuracy = sum(result_val) / len(result_val)
#         logger.info(f'Validation accuracy: {val_accuracy:.2%}')
#         return val_accuracy
#     return 0.0


def train_agent(args, actor):
    """Train the agent with optional validation at checkpoints"""
    if args.do_train:
        prepare_train(args, actor)
        actor.train_loop()
        
        # best_val_accuracy = 0.0
        # validation_results = []
        
        # while actor.train_iter < args.max_train_iter:
        #     actor.train_step()
        #     actor.train_iter += 1
            
        #     # Save checkpoint if needed
        #     if args.save_every and actor.train_iter % args.save_every == 0:
        #         dump_json(
        #             {'train_iter': actor.train_iter},
        #             f"{args.result_dir}/train_ckpt_info.json"
        #         )
            
        #     # Run validation if enabled
        #     if args.do_validation and actor.train_iter % args.validation_interval == 0:
        #         val_accuracy = validate_agent(args, actor)
        #         validation_results.append({
        #             'iteration': actor.train_iter,
        #             'accuracy': val_accuracy
        #         })
                
        #         # Save validation results
        #         dump_json(
        #             validation_results,
        #             f"{args.result_dir}/validation_results.json",
        #             indent=4
        #         )
                
        #         # Save best model if improved
        #         if val_accuracy > best_val_accuracy:
        #             best_val_accuracy = val_accuracy
        #             # Copy current checkpoint to best checkpoint
        #             if os.path.exists(args.ckpt_dir):
        #                 best_ckpt_dir = args.ckpt_dir + '_best'
        #                 shutil.copytree(args.ckpt_dir, best_ckpt_dir, dirs_exist_ok=True)
        #                 logger.info(f'Saved best model with validation accuracy: {val_accuracy:.2%}')


# test related
def evaluate_agent(args, actor, dataset_registry_updates=None):
    """
    Evaluates the performance of an agent using the provided arguments and actor.
    Args:
        args (Namespace): A namespace object containing various arguments and configurations.
            - do_test (bool): Flag to indicate whether to perform testing.
            - test_dataset_name (str, optional): Name of the test dataset to use.
            - dataset_name (str): Name of the dataset to use (legacy compatibility)
        actor (object): The agent or model to be evaluated.
        dataset_registry_updates (dict, optional): Updates to be applied to the dataset registry.
    """
    if args.do_test or args.do_val:
        if args.test_dataset_name:
            args.dataset_name = args.test_dataset_name
        kwargs = vars(args)
        
        if dataset_registry_updates:
            dataset_registry.update(dataset_registry_updates)
        data_pipeline = data_pipeline_factory(train=args.do_val, **kwargs)

        eval_manager = EvalManager(args, actor, data_pipeline)
        eval_manager.test_loop()

# cleanup
def cleanup(args):
    """
    Cleans up by copying checkpoint directory

    Args:
        args: An object containing configuration attributes
    """
    if args.standalone_ckpt:
        shutil.copytree(args.ckpt_dir, args.result_dir + '/' + args.ckpt_name, dirs_exist_ok=True)
    print(f'Done. result dir is {args.result_dir}')
