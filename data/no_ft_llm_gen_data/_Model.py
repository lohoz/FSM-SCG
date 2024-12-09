import os
from utils.data_utils import data_utils
from utils.prompt_utils import prompt_utils
from utils.Model import Model


class _Model(Model):

    def __init__(self, config: str):
        super().__init__(config)


    def generate_no_fsm(self, user_requirement: str, version: str, output_path: str):
        prompt = prompt_utils.generate_code_no_fsm_prompt(user_requirement, version)
        response = self.single_dialogue(prompt)

        data = {}
        data['user_requirement'] = user_requirement
        data['code'] = response
        data['model'] = self.model
        data_utils.append_jsonl(output_path, data)


    def generate_use_fsm(self, user_requirement: str, version: str, output_path: str):
        conversation = []
        prompt_1, prompt_2 = prompt_utils.generate_code_with_fsm(user_requirement, version)

        response_1, conversation = self.multiple_dialogue(prompt_1, conversation, random_parameters=True)
        response_2, conversation = self.multiple_dialogue(prompt_2, conversation, random_parameters=True)

        data = {}
        data['user_requirement'] = user_requirement
        data['FSM'] = response_1
        data['code'] = response_2
        data['model'] = self.model
        data_utils.append_jsonl(output_path, data)




class Evaluation_gen:
    @staticmethod
    def generate_code_for_effectiveness_and_security(model: _Model, evaluation_path: str, is_use_fsm: bool):
        output_path = ''
        if is_use_fsm:
            output_path = model.model + '_use_fsm_for_effectiveness_and_security.jsonl'
        else:
            output_path = model.model + '_no_fsm_for_effectiveness_and_security.jsonl'

        dataset = data_utils.load_jsonl_dataset(evaluation_path)
        
        user_requirement_list = []
        for data in dataset:
            user_requirement_list.append(data['user_requirement'])

        for data in dataset:
            if is_use_fsm:
                model.generate_use_fsm(user_requirement=data['user_requirement'], version=data['version'], output_path=output_path)
            else:
                model.generate_no_fsm(user_requirement=data['user_requirement'], version=data['version'], output_path=output_path)
    

    @staticmethod
    def generate_code_for_correctness(model: _Model, evaluation_path: str, is_use_fsm: bool, n: int):
        output_path = ''
        if is_use_fsm:
            output_path = model.model + '_use_fsm_for_correctness.jsonl'
        else:
            output_path = model.model + '_no_fsm_for_correctness.jsonl'

        dataset = data_utils.load_jsonl_dataset(evaluation_path)

        user_requirement_list = []
        for data in dataset:
            user_requirement_list.append(data['user_requirement'])

        test_record_list = [n for i in range(len(user_requirement_list))]

        if os.path.exists(output_path):
            current_data = data_utils.load_jsonl_dataset(output_path)

            for data in current_data:
                task_id = data['task_id']
                test_record_list[task_id] -= 1
            print(test_record_list)


        def is_all_zero(test_record_list: list):
            for test_record in test_record_list:
                if test_record != 0:
                    return False
            return True
        
        for i in range(0, n):
            if is_all_zero(test_record_list):
                break

            for j, user_requirement in enumerate(user_requirement_list):
            
                if test_record_list[j] > 0:
                    if is_use_fsm:
                        model.generate_use_fsm(user_requirement=user_requirement, version='0.8.0', output_path=output_path)
                    else:
                        model.generate_no_fsm(user_requirement=user_requirement, version='0.8.0', output_path=output_path)
                    test_record_list[j] -= 1
                
                if is_all_zero(test_record_list):
                    break