import logging

from statistics import mean

from cognitive_base.utils import dump_json, load_json
from cognitive_base.utils.log import move_log_file, construct_task_folder

logger = logging.getLogger("logger")


class EvalManager:
    def __init__(self, args, actor, data_pipeline):
        self.args = args

        self.actor = actor
        # parallel need multiple actors
        # self.actors = actors

        self.dataset_name = args.dataset_name
        self.data_pipeline = data_pipeline
        dataset, dataloader = self.data_pipeline.get_dataloader()
        self.dataset = dataset
        self.dataloader = dataloader

        self.result_dir = args.result_dir

        self.result_d = {}
        if self.args.resume:
            print(f"\033[35mLoading result_dict\033[0m")
            self.result_d = load_json(f"{self.result_dir}/result_dict.json")

    def set_actor_attr(self, actor):
        if not self.args.use_public_tests:
            actor.max_task_attempts = 1
            actor.critic_module.use_critic = False

        actor.critic_module.use_critic_success = False
        actor.train = False

    def test_loop_serial(self):
        actor = self.actor
        self.set_actor_attr(actor)

        n_test = len(self.dataloader)
        for i, batch in enumerate(self.dataloader):
            # use this instead of dataloader to break because need full dataloader for eval later
            if self.args.max_test_iter and i >= self.args.max_test_iter:
                break
            full_task = list(batch)[0] if self.args.dataset_type == 'pytorch' else batch
            full_task = self.data_pipeline.preprocess(full_task)
            task_id = str(full_task['task_id'])
            if task_id in self.result_d:
                continue
            logger.info(f'[test iter]: {i + 1}/{n_test}\n')

            success, parsed_result = actor.test_one(full_task)

            if parsed_result:
                self.result_d[task_id] = success
                result_val = self.result_d.values()
                logger.info(f'acc:{sum(result_val)}/{len(self.result_d)} = {mean(result_val):.2%}')
                dump_json(self.result_d, f"{self.result_dir}/result_dict.json", indent=4)

            task_folder = construct_task_folder(self.result_dir, 'test', task_id)
            move_log_file(f"{task_folder}/logfile.log", self.result_dir)

    def test_loop(self):
        if self.args.num_agents == 1:
            self.test_loop_serial()
        elif self.args.num_agents > 1:
            # self.test_loop_parallel()
            raise NotImplementedError("Parallel eval not implemented")

        # Postprocess, for batch eval
        if self.args.eval_later:
            self.data_pipeline.postprocess(dataloader=self.dataloader, dataset=self.dataset, result_dir=self.result_dir)
