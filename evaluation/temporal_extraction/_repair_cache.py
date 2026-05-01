import json
import os

fp = "/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/temporal_extraction/cache/embedding_cache.json"
with open(fp) as f:
    text = f.read()
print("Original size:", len(text))
dec = json.JSONDecoder()
obj1, idx1 = dec.raw_decode(text)
print("First object:", len(obj1), "keys, ends at", idx1)
with open(fp, "w") as f:
    json.dump(obj1, f)
print("Wrote back; new size:", os.path.getsize(fp))
