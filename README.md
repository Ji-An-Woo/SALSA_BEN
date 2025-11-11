SALSA-repro (connected) - runs real SALSA (when LWE-benchmarking repo is cloned)
------------------------------------------------------------------------------
What this bundle does:
 - Precompute small datasets (n=10 and n=30) for baseline and your obfuscated secret
 - Wrapper 'src/run_salsa_connected.py' will call the real SALSA entrypoint:
     train_and_recover.py from https://github.com/facebookresearch/LWE-benchmarking
   (You must clone that repo into 'external/LWE-benchmarking')
 - Captures stdout of the SALSA run, saves it to results/, and tries to extract predicted secrets
 - Records all parameters and outputs to CSVs and plots

Important references:
 - LWE-benchmarking repository (contains SALSA code and train_and_recover.py). See README and SALSA usage examples. citeturn2view0

Quick steps:
 1) Clone external repo: git clone https://github.com/facebookresearch/LWE-benchmarking external/LWE-benchmarking
 2) Create conda env as described in that repo (it may require NVIDIA GPU and dependencies)
 3) Install this bundle requirements: pip install -r requirements.txt
 4) Generate datasets: python src/data_gen_obfuscate_fixed.py
 5) Run SALSA connected wrapper: python src/run_salsa_connected.py
 6) Evaluate: python src/evaluate_and_plot.py

Notes:
 - SALSA training is resource intensive. For n=10 and n=30 this should be lightweight but still may require GPU for transformer models.
 - If external repo's entrypoint location differs, edit 'EXTERNAL_TRAIN_SCRIPT' variable in src/run_salsa_connected.py
