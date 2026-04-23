import json
import argparse

def average_input_tokens(data):
    total_tokens = 0
    total_items = 0

    for category, items in data.get("results", {}).items():
        for item in items:
            total_tokens += item.get("input_tokens", 0)
            total_items += 1

    return total_tokens / total_items if total_items else 0


def main():
    parser = argparse.ArgumentParser(description="Compute average input_tokens from JSON")
    parser.add_argument("file", help="Path to JSON file")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        data = json.load(f)

    avg = average_input_tokens(data)
    print(f"Average input_tokens: {avg:.2f}")


if __name__ == "__main__":
    main()
