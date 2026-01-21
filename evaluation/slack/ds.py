import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset

# ds = load_dataset("spencer/software_slacks")

# df = pd.DataFrame(ds["train"])
# df_unique = df.drop_duplicates()

# ds = Dataset.from_pandas(df_unique)

# # ts_ordered = ds.filter(lambda x: x["workspace"] == "pythondev").sort("ts")
# # print(ts_ordered[-1])
# workspace = ds.filter(lambda x: x["workspace"] == "pythondev")
# lengths = workspace.map(lambda x: {"length": len(x["text"])})
# average_length = np.median(lengths["length"])
# print(average_length)

ds = load_dataset("unionai/flyte-slack-data")
df = pd.DataFrame(ds["train"])
df_unique = df.drop_duplicates()

ds = Dataset.from_pandas(df_unique)
lengths = ds.map(lambda x: {"length": len(x["input"])})
average_length = np.mean(lengths["length"])
print(average_length)
