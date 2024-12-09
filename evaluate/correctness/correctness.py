import os
from utils.data_utils import data_utils
from utils.solidity_utils import solidity_utils
import subprocess
import argparse
from colorama import Fore, init
import re
from scipy.special import comb
import numpy as np
import itertools



def extract_test_results(output):
    passing_matches = re.search(r"(\d+)\s+passing", output)
    failing_matches = re.search(r"(\d+)\s+failing", output)
    
    passing_count = int(passing_matches.group(1)) if passing_matches else 0
    failing_count = int(failing_matches.group(1)) if failing_matches else 0
    
    return passing_count, failing_count



def test_all_by_hardhat(smart_contract, test_info, hardhat_test_path):
    passing_count, failing_count = 0, 0
    try: 
        data_utils.save_to_file(hardhat_test_path + '/contracts/contract.sol', smart_contract)

        test_list = test_info['test']
        for i, test_content in enumerate(test_list):
            data_utils.save_to_file(f'{hardhat_test_path}/test/test-{i}.js', test_content['test_code'])

        version = solidity_utils.extract_solc_version(smart_contract)

        version_config_text = f"""require("@nomiclabs/hardhat-waffle");\nmodule.exports = {{\n  solidity: "{version}",\n}};"""

        # Change the version of Hardhat configuration
        data_utils.save_to_file(hardhat_test_path + '/hardhat.config.js', version_config_text)

        try:
            # Run Hardhat Test
            test_result = subprocess.run(
                ['npx', 'hardhat', 'test'],
                cwd=hardhat_test_path,
                capture_output=True,
                text=True
            )
            # print("Test output:", test_result.stdout)
            
        except subprocess.CalledProcessError as se:
            print("Error during compilation or testing:", se.stderr)
        
        passing_count, failing_count = extract_test_results(test_result.stdout)
        print(Fore.GREEN + f"Passing cases: {passing_count}")
        print(Fore.RED + f"Failing cases: {failing_count}")
    
    except Exception as e:
        print('Error:', str(e))

    finally: 
        # Delete temporary files
        data_utils.delete_file(hardhat_test_path + '/contracts/contract.sol')
        for i in range(0, len(test_list)):
            data_utils.delete_file(f'{hardhat_test_path}/test/test-{i}.js')
    
    return {'test_id': test_info['test_id'], 'passing_count': passing_count, 'failing_count': failing_count}



def test_detail_by_hardhat(smart_contract, test_info, hardhat_test_path):
    detail_info = []
    try:
        data_utils.save_to_file(hardhat_test_path + '/contracts/contract.sol', smart_contract)

        version = solidity_utils.extract_solc_version(smart_contract)

        version_config_text = f"""require("@nomiclabs/hardhat-waffle");\nmodule.exports = {{\n  solidity: "{version}",\n}};"""

        # Change the version of Hardhat configuration
        data_utils.save_to_file(hardhat_test_path + '/hardhat.config.js', version_config_text)


        test_codes = test_info['test_codes']
        for test_code in test_codes:

            data_utils.save_to_file(hardhat_test_path + '/test/test.js', test_code['code'])

            stdout = ''
            try:
                # Run Hardhat Test
                test_result = subprocess.run(
                    ['npx', 'hardhat', 'test'],
                    cwd=hardhat_test_path,
                    capture_output=True,
                    text=True
                )
                # print("Test output:", test_result.stdout)
                stdout = test_result.stdout
            except subprocess.CalledProcessError as se:
                print("Error during compilation or testing:", se.stderr)
            
            finally:
                passing_count, failing_count = extract_test_results(stdout)
                print(Fore.CYAN + f"### {test_code['test']}:")
                print(Fore.GREEN + f"Passing cases: {passing_count}")
                print(Fore.RED + f"Failing cases: {failing_count}")  

                data = {'test': test_code['test'], 'passing_count': passing_count, 'failing_count': failing_count}
                detail_info.append(data)
    
    except subprocess.CalledProcessError as se:
        print("Error during compilation or testing:", se.stderr)
    
    finally: 
        # Delete temporary files
        data_utils.delete_file(hardhat_test_path + '/contracts/contract.sol')
        data_utils.delete_file(hardhat_test_path + '/test/test.js')

    return {'task_id': test_info['task_id'], 'detail_info': detail_info}



def evaluate_correctness(data_path, result_path: str, 
                         benchmark_path: str,
                         hardhat_test_path: str, 
                         remove_import_statements: bool,
                         openzeppelin_path: str):
    dataset = data_utils.load_jsonl_dataset(data_path)

    benchmark = data_utils.load_jsonl_dataset(benchmark_path)

    test_result = []
    for data in dataset:
        code = data_utils.extract_code(data['code'])

        if remove_import_statements:
            code = data_utils.remove_import_statements(code)
        else:
            code = code.replace('@openzeppelin', openzeppelin_path)

        test_info = benchmark[data['task_id']] 
        
        detail_info = test_detail_by_hardhat(code, test_info, hardhat_test_path)
        test_result.append(detail_info)
    
    data_utils.append_jsonl(result_path, {'file': data_path, 'test_result': test_result})
        

# ------------------------------------------------------------------------------------------------------------

def cal_pass_at_k(n, k, k_success):
    total_combinations = comb(k, n)
    if k - k_success >= n:
        without_k_success_combinations = comb(k - k_success, n)
    else:
        without_k_success_combinations = 0

    with_k_success_combinations = total_combinations - without_k_success_combinations

    pass_at_k = with_k_success_combinations / total_combinations

    return pass_at_k


 # --------------------------------------------------------------------------------------------------------
def estimator(n: int, c: int, k: int) -> float:
    """Calculates 1 - comb(n - c, k) / comb(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
 # --------------------------------------------------------------------------------------------------------



def compute_pass_at_k(out_path, result_info, k=[1, 3, 5, 10]):
    file = result_info['file']
    test_result_list = result_info['test_result']
    
    entire_contract_level_test_result = {}
    multi_function_level_test_result = {}
    for test_result in test_result_list:
        task_id = test_result['task_id']
        
        if task_id not in entire_contract_level_test_result:
            entire_contract_level_test_result[task_id] = {'success': 0, 'fail': 0}
            multi_function_level_test_result[task_id] = {}

        detail_info = test_result['detail_info']

        is_contract_fail = False
        for one_info in detail_info:
            test = one_info['test']

            if test not in multi_function_level_test_result[task_id]:
                multi_function_level_test_result[task_id][test] = {'success': 0, 'fail': 0}

            if not (one_info['failing_count'] == 0 and one_info['passing_count'] != 0):
                # If a functional test fails, it is considered a complete failure
                is_contract_fail = True

                multi_function_level_test_result[task_id][test]['fail'] += 1
            else:
               multi_function_level_test_result[task_id][test]['success'] += 1

        
        if is_contract_fail:
            entire_contract_level_test_result[task_id]['fail'] += 1
            continue 
        else:
            # Pass all tests
            entire_contract_level_test_result[task_id]['success'] += 1
    
    # print(entire_contract_level_test_result)
    # print(multi_function_level_test_result)

    #-----------------------compute entire contract level pass@k-----------------------------------------------
    def compute_entire_contract_level_pass_at_k(k: int):
        total_entire_contract_level_pass_at_k = 0
        for task_id, result in entire_contract_level_test_result.items():
            pass_at_k = estimator(result['success'] + result['fail'], result['success'], k)
            total_entire_contract_level_pass_at_k += pass_at_k

        average_entire_contract_level_pass_at_k = total_entire_contract_level_pass_at_k / len(entire_contract_level_test_result) if len(entire_contract_level_test_result) > 0 else 0
        # print(f'average_entire_contract_level_pass_at_k: {Fore.GREEN}{average_entire_contract_level_pass_at_k}')
        return average_entire_contract_level_pass_at_k
    
    ks = k
    entire_contract_level_pass_at_k_list = {f"pass@{k}": float(compute_entire_contract_level_pass_at_k(k)) for k in ks}
    #-----------------------------------------------------------------------------------

    #-----------------------compute multi function level pass@k------------------------------------------
    def compute_multi_function_level_pass_at_k(k: int):
        total_multi_function_level_pass_at_k = 0
        functionality_count = 0
        for task_id, functionality_result in multi_function_level_test_result.items():
            for functionality, result in functionality_result.items():
                pass_at_k = estimator(result['success'] + result['fail'], result['success'], k)
                # print(pass_at_k)
                total_multi_function_level_pass_at_k += pass_at_k
                functionality_count += 1

        average_multi_function_level_pass_at_k = total_multi_function_level_pass_at_k / functionality_count if functionality_count > 0 else 0
        # print(f'average_functionality_pass_at_k: {Fore.GREEN}{average_functionality_pass_at_k}')
        return average_multi_function_level_pass_at_k
    
    ks = k
    multi_function_level_pass_at_k_list = {f"pass@{k}": float(compute_multi_function_level_pass_at_k(k)) for k in ks}
    #-----------------------------------------------------------------------------------

    data_utils.append_jsonl(out_path, {'file': file, 'entire_contract_level_pass_at_k': entire_contract_level_pass_at_k_list, 'multi_function_level_pass_at_k': multi_function_level_pass_at_k_list})

    print(Fore.YELLOW + file)
    print(f'{Fore.GREEN}{entire_contract_level_pass_at_k_list}')
    print(f'{Fore.GREEN}{multi_function_level_pass_at_k_list}')
    return entire_contract_level_pass_at_k_list, multi_function_level_pass_at_k_list
            


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze validity of generated code."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Please enter the data path here.",
    )

    parser.add_argument(
        "--benchmark_path",
        type=str,
        default="./smart_contract_benchmark.jsonl",
        help="Please enter benchmark path here.",
    )

    parser.add_argument(
        "--hardhat_test_path",
        type=str,
        default="./hardhat_test",
        help="Please enter hardhat test path here.",
    )

    parser.add_argument(
        "--detail_result_path",
        type=str,
        default="",
        help="Please enter detail result path here.",
    )

    parser.add_argument(
        "--pass_at_k_result_path",
        type=str,
        default="",
        help="Please enter pass@k result path here.",
    )

    parser.add_argument(
        "--remove_import_statements",
        type=bool,
        default=False,
        help="Whether to remove import statements.",
    )

    parser.add_argument(
        "--openzeppelin_path",
        type=str,
        default="./openzeppelin",
        help="Path to the OpenZeppelin library.",
    )

    return parser.parse_args()



def main(args):
    init(autoreset=True)

    if not args.data_path:
        print("please enter data path")
        return
    
    if args.detail_result_path == '':
        file_name = args.data_path.split('/')[-1]
        result_folder = args.data_path.replace(file_name, '') + 'evaluation_result/'
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        args.detail_result_path = result_folder + file_name.split('.')[0] + '_correctness_detail_result.jsonl'
    
    evaluate_correctness(args.data_path, 
                         args.detail_result_path, 
                         args.benchmark_path, 
                         args.hardhat_test_path, 
                         args.remove_import_statements, 
                         args.openzeppelin_path)

    if args.pass_at_k_result_path == '':
        file_name = args.data_path.split('/')[-1]
        result_folder = args.data_path.replace(file_name, '') + 'evaluation_result/'
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        args.pass_at_k_result_path = result_folder + file_name.split('.')[0] + '_correctness_pass_at_k_result.jsonl'

    result_dataset = data_utils.load_jsonl_dataset(args.detail_result_path)
    for result in result_dataset:
        compute_pass_at_k(args.pass_at_k_result_path, result)



if __name__ == "__main__":
    args = parse_args()
    main(args)