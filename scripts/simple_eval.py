import json
import argparse


def main(args):

    vlm = args.vlm
    llm = args.llm

    results_path = f'{vlm}_{llm}.json'

    # Load the data
    with open(results_path, 'r') as f:
        data = json.load(f)

    # Counters
    total = 0
    correct = 0

    a_total = 0
    a_correct = 0

    b_total = 0
    b_correct = 0

    # Evaluation
    for item in data:
        uid = item.get("uid", "")
        answer = item.get("answer")

        total += 1

        if uid.endswith("A"):
            a_total += 1
            if answer == "positive":
                correct += 1
                a_correct += 1
        elif uid.endswith("B"):
            b_total += 1
            if answer == "negative":
                correct += 1
                b_correct += 1

    # Accuracy calculations
    overall_accuracy = (correct / total) * 100 if total > 0 else 0
    a_accuracy = (a_correct / a_total) * 100 if a_total > 0 else 0
    b_accuracy = (b_correct / b_total) * 100 if b_total > 0 else 0

    # Output
    print(f"Results file: {results_path}")
    print(f"Total records: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {overall_accuracy:.2f}%\n")
    print(f"Query ID 'A' accuracy: {a_correct}/{a_total} ({a_accuracy:.2f}%)")
    print(f"Query ID 'B' accuracy: {b_correct}/{b_total} ({b_accuracy:.2f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm', type=str, choices=['blip2', 'instructBLIP'], help='choose a caption model')
    parser.add_argument('--llm', type=str, choices=['llama', 'deepseek', 'gpt41'], help='choose a LLM model')

    args = parser.parse_args()
    main(args)
