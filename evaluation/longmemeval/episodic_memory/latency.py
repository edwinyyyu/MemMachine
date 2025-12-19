import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("--data-path", required=True)

args = parser.parse_args()
data_path = args.data_path

with open(data_path, "r") as f:
    data = json.load(f)

total_latency = 0
for item in data:
    total_latency += item["memory_latency"]

print(total_latency / 500)
