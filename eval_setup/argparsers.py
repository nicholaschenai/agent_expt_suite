from cognitive_base.utils.argparsers import get_base_parser


def get_base_agent_parser():
    """
    parse args that are more agent-based n experiment-based
    """
    parser = get_base_parser()

    # dataset
    parser.add_argument("--train_dataset_name", type=str, default="")
    parser.add_argument("--test_dataset_name", type=str, default="MBPP_Plus")
    parser.add_argument("--dataset_type", type=str, default="pytorch", help="which framework's dataset. pytorch or hf")
    parser.add_argument("--max_len", type=int, default=36000, help="max len of train example for ctx len mgmt")

    # train
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--max_attempts_per_task", type=int, default=4, help="rollout length")
    parser.add_argument("--max_train_iter", type=int, default=100, help="training steps")
    parser.add_argument("--load_train", action="store_true", help="load trainset data")

    # eval
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--eval_later", action="store_true", help="batch eval at the end instead of every step")
    parser.add_argument("--use_public_tests", action="store_true", help="use public tests before official eval")
    parser.add_argument("--max_test_iter", type=int, default=0, help="break after k test iter, for debugging")
    
    # saving / checkpointing
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--standalone_ckpt", action="store_true", help="set ckpt to folder independent of results")
    parser.add_argument("--ckpt_name", type=str, default="ckpt")
    parser.add_argument("--ckpt_dir", type=str, default="", help="CURRENT checkpoint")
    parser.add_argument("--clone_ckpt", type=str, default="", help="clone from a checkpoint")
    parser.add_argument("--save_every", type=int, default=10, help="save separate checkpoints every k train steps")
    
    # strategy
    parser.add_argument("--agent_type", type=str, default="coala")
    parser.add_argument("--retrieval_top_k", type=int, default=5)

    # debug
    parser.add_argument("--debug_subset", type=int, default=50, help="subset dataset for debugging")
    parser.add_argument("--max_display_tests", type=int, default=10, help="max num of env outputs to display")
    parser.add_argument("--max_display_chars", type=int, default=1000, help="max len of env outputs to display")

    # parallel
    parser.add_argument("--parallel_api", action="store_true", help="parallel api calls if possible")
    parser.add_argument("--num_agents", type=int, default=1, help="multi agents for parallel eval only")

    return parser
