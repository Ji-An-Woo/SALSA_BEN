import pickle
import json

# Check params.pkl
try:
    with open('data/precomputed/baseline_n10_binary/params.pkl', 'rb') as f:
        params = pickle.load(f)
    print("✅ params.pkl contents:")
    for key, value in params.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"❌ Error reading params.pkl: {e}")

# Check meta.json
try:
    with open('data/precomputed/baseline_n10_binary/meta.json', 'r') as f:
        meta = json.load(f)
    print("\n✅ meta.json contents:")
    print(json.dumps(meta, indent=2))
except Exception as e:
    print(f"❌ Error reading meta.json: {e}")

# Check original dataset params for comparison
try:
    with open('data/precomputed/baseline_n10/params.pkl', 'rb') as f:
        orig_params = pickle.load(f)
    print("\n✅ Original baseline_n10 params.pkl contents:")
    for key, value in orig_params.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"❌ Error reading original params.pkl: {e}")