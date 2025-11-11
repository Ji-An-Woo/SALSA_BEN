import numpy as np
import os, json

def save_json(p, obj):
    os.makedirs(os.path.dirname(str(p)), exist_ok=True)
    with open(str(p), "w") as f:
        json.dump(obj, f, indent=2)

def load_json(p):
    with open(p, "r") as f:
        import json
        return json.load(f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
