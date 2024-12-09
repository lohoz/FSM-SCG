import os
from utils.data_utils import data_utils
from utils.prompt_utils import prompt_utils
from utils.solidity_utils import solidity_utils
from utils.fsm_utils import fsm_utils
from evaluate.security import slither_check
from utils.Model import Model


class _Model(Model):

    def __init__(self, config: str):
        super().__init__(config)



    def generate_use_fsm_scg(self, user_requirement: str, version: str, openzeppelin_path: str, output_path: str, feedback_count: int=1):
        conversation = []
        prompt_1, prompt_2 = prompt_utils.generate_code_with_fsm(user_requirement, version)

        # Initial FSM generation
        response_1, conversation = self.multiple_dialogue(prompt_1, conversation, random_parameters=True)

        # FSM validation and feedback loop
        fsm = data_utils.extract_fsm(response_1)
        response_1 = self._validate_and_regenerate_fsm(fsm, feedback_count, conversation)
    
        # Check FSM reachability and cycle properties
        fsm = data_utils.extract_fsm(response_1)
        response_1 = self._check_reachability_and_cycles(fsm, feedback_count, conversation)

        # Generate smart contract code
        response_2, conversation = self.multiple_dialogue(prompt_2, conversation, random_parameters=True)

        # Smart contract code compilation and feedback loop
        code = data_utils.extract_code(response_2)
        code = code.replace('@openzeppelin', openzeppelin_path)
        response_2 = self._compile_and_validate_code(code, feedback_count, conversation, openzeppelin_path)

        # Security check and feedback loop
        response_2 = self._check_security_risks(response_2, feedback_count, conversation, openzeppelin_path)


        data = {}
        data['user_requirement'] = user_requirement
        data['FSM'] = data_utils.extract_fsm(response_1)
        data['code'] = data_utils.extract_code(response_2)
        data['model'] = self.model
        data_utils.append_jsonl(output_path, data)



    def _validate_and_regenerate_fsm(self, fsm, feedback_count, conversation):
        fb_count = feedback_count
        while True:
            is_valid, message = fsm_utils.validate_fsm(fsm)
            if is_valid or fb_count <= 0:
                break
            prompt = f'The generated FSM has the following issues, please regenerate the FSM:\n{message}'
            response, conversation = self.multiple_dialogue(prompt, conversation, random_parameters=True)
            fsm = data_utils.extract_fsm(response)
            fb_count -= 1
        return response



    def _check_reachability_and_cycles(self, fsm, feedback_count, conversation):
        fb_count = feedback_count
        while True:
            unreachable_states, has_cycle = fsm_utils.check_reachability_and_cycles(fsm)
            if (not unreachable_states and has_cycle) or fb_count <= 0:
                break
            prompt = 'The generated FSM has the following issues, please regenerate the FSM:'
            if unreachable_states:
                prompt += f'\n### List of unreachable states: {unreachable_states}'
            if not has_cycle:
                prompt += '\n### The graph composed of states does not have cycles'
            response, conversation = self.multiple_dialogue(prompt, conversation, random_parameters=True)
            fsm = data_utils.extract_fsm(response)
            fb_count -= 1
        return response
    


    def _compile_and_validate_code(self, code, feedback_count, conversation, openzeppelin_path):
        fb_count = feedback_count
        while True:
            compile_result, compile_info = solidity_utils.compile_solidity(code)
            if compile_result or fb_count <= 0:
                break
            compile_error_prompt = prompt_utils.feedback_by_compile_error_prompt(
                solidity_utils.extract_solcx_compile_error(str(compile_info))
            )
            response, conversation = self.multiple_dialogue(compile_error_prompt, conversation, random_parameters=True)
            code = data_utils.extract_code(response)
            code = code.replace('@openzeppelin', openzeppelin_path)
            fb_count -= 1
        return response



    def _check_security_risks(self, response, feedback_count, conversation, openzeppelin_path):
        fb_count = feedback_count
        while True:
            code = data_utils.extract_code(response)
            code = code.replace('@openzeppelin', openzeppelin_path)
            check_info = slither_check.check_one_by_slither(code)
            if isinstance(check_info, str) or len(check_info) == 0 or fb_count <= 0:
                break
            security_risk_prompt = prompt_utils.feedback_by_security_risk_prompt(check_info)
            response, conversation = self.multiple_dialogue(security_risk_prompt, conversation, random_parameters=True)
            fb_count -= 1
        return response






class Evaluation_gen:
    @staticmethod
    def generate_code_for_effectiveness_and_security(model: _Model, evaluation_path: str, openzeppelin_path: str):
        output_path = model.model + '_use_fsm-scg_for_effectiveness_and_security.jsonl'

        dataset = data_utils.load_jsonl_dataset(evaluation_path)
        
        user_requirement_list = []
        for data in dataset:
            user_requirement_list.append(data['user_requirement'])

        for data in dataset:
            model.generate_use_fsm_scg(user_requirement=data['user_requirement'], version=data['version'], openzeppelin_path=openzeppelin_path, output_path=output_path)
    

    @staticmethod
    def generate_code_for_correctness(model: _Model, evaluation_path: str, openzeppelin_path: str, n: int):
        output_path = model.model + '_use_fsm-scg_for_correctness.jsonl'

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
                    model.generate_use_fsm_scg(user_requirement=user_requirement, version='0.8.0', openzeppelin_path=openzeppelin_path, output_path=output_path)
                    test_record_list[j] -= 1
                
                if is_all_zero(test_record_list):
                    break