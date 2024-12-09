from _Model import _Model, Evaluation_gen
from colorama import init, Fore
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="The fine tuned Qwen generates data for evaluation."
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
    
    parser.add_argument(
        "--openzeppelin_path",
        type=str,
        default="./openzeppelin",
        help="Path to the OpenZeppelin library.",
    )

    return parser.parse_args()


def main(args):
    init(autoreset=True)
    qwen = _Model('qwen')

    if args.evaluation_type == 'effectiveness' or args.evaluation_type == 'security':
        Evaluation_gen.generate_code_for_effectiveness_and_security(qwen, args.evaluation_path, args.openzeppelin_path)
    elif args.evaluation_type == 'correctness':
        Evaluation_gen.generate_code_for_correctness(qwen, args.evaluation_path, args.openzeppelin_path, argparse.n)
    else:
        raise Fore.RED + 'The type of evaluation is incorrect.'


if __name__ == '__main__':
    args = parse_args()
    main(args)