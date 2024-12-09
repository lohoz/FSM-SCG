import os
from utils.data_utils import data_utils
from utils.prompt_utils import prompt_utils
from _Model import Evaluation_gen
from colorama import Fore, Back, Style, init
import google.generativeai as genai
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="GeMini generates data for evaluation."
    )

    parser.add_argument(
        "--evaluation_path",
        type=str,
        default="evaluation.jsonl",
        help="Path to the evaluation data file.",
    )

    parser.add_argument(
        "--evaluation_type",
        type=str,
        default="effectiveness",
        help="The types of evaluations include effectiveness, security and correctness.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="If the evaluation type is correctness, it is necessary to set the number of code samples generated for each requirement",
    )

    parser.add_argument(
        "--is_use_fsm",
        type=bool,
        default=False,
        help="Whether to use fsm.",
    )

    return parser.parse_args()

model = ''

def generate_no_fsm(gemini: genai.GenerativeModel, user_requirement: str, version: str, output_path: str):
    prompt = 'You are an expert in smart contract programming. ' + prompt_utils.generate_code_no_fsm_prompt(user_requirement, version)

    response = gemini.generate_content(prompt)
    try:
        res = response.text
    except Exception:
        res = ""
    print(Fore.GREEN + res)


    data = {}
    data['user_requirement'] = user_requirement
    data['code'] = res
    data['model'] = model
    data_utils.append_jsonl(output_path, data)

        

def generate_use_fsm(gemini: genai.GenerativeModel, user_requirement: str, version: str, output_path: str):
    prompt_1, prompt_2 = prompt_utils.generate_code_with_fsm(user_requirement, version)

    prompt_1 = 'You are an expert in smart contract programming. ' + prompt_1

    chat = gemini.start_chat(history=[])

    response_1 = chat.send_message(prompt_1).text
    print(Fore.GREEN + response_1)

    response_2 = chat.send_message(prompt_2).text
    print(Fore.GREEN + response_2)

    data = {}
    data['user_requirement'] = user_requirement
    data['FSM'] = response_1
    data['code'] = response_2
    data['model'] = model
    data_utils.append_jsonl(output_path, data)



def generate_code_for_effectiveness_and_security(gemini: genai.GenerativeModel, evaluation_path: str, is_use_fsm: bool):
        output_path = ''
        if is_use_fsm:
            output_path = model + '_use_fsm_for_effectiveness_and_security.jsonl'
        else:
            output_path = model + '_no_fsm_for_effectiveness_and_security.jsonl'

        dataset = data_utils.load_jsonl_dataset(evaluation_path)
        
        user_requirement_list = []
        for data in dataset:
            user_requirement_list.append(data['user_requirement'])

        for data in dataset:
            if is_use_fsm:
                generate_use_fsm(gemini, user_requirement=data['user_requirement'], version=data['version'], output_path=output_path)
            else:
                generate_no_fsm(gemini, user_requirement=data['user_requirement'], version=data['version'], output_path=output_path)



def generate_code_for_correctness(gemini: genai.GenerativeModel, evaluation_path: str, is_use_fsm: bool, n: int):
    output_path = ''
    if is_use_fsm:
        output_path = model + '_use_fsm_for_correctness.jsonl'
    else:
        output_path = model + '_no_fsm_for_correctness.jsonl'

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
                    generate_use_fsm(gemini, user_requirement=user_requirement, version='0.8.0', output_path=output_path)
                else:
                    generate_no_fsm(gemini, user_requirement=user_requirement, version='0.8.0', output_path=output_path)
                test_record_list[j] -= 1
            
            if is_all_zero(test_record_list):
                break


def main(args):
    init(autoreset=True)
    config = data_utils.load_config('../../config/llm_api.config')
    global model
    model = config['gemini']['model']
    key = config['gemini']['key']
    genai.configure(api_key=key, transport='rest')
    gemini = genai.GenerativeModel(model)


    if args.evaluation_type == 'effectiveness' or args.evaluation_type == 'security':
        Evaluation_gen.generate_code_for_effectiveness_and_security(gemini, args.evaluation_path, args.is_use_fsm)
    elif args.evaluation_type == 'correctness':
        Evaluation_gen.generate_code_for_correctness(gemini, args.evaluation_path, args.is_use_fsm, argparse.n)
    else:
        raise Fore.RED + 'The type of evaluation is incorrect.'


if __name__ == '__main__':
    args = parse_args()
    main(args)