import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import RunLog, run_consistency, write_json
from science import EXPERIMENT_DIR, load_clean_inputs, scientific_source_hash, source_file_hashes


def main():
    clean = load_clean_inputs(os.path.join(EXPERIMENT_DIR, "inputs", "clean_signals.npz"))
    source_hash = scientific_source_hash(clean["continuous_source_sha256"], source_file_hashes())
    log = RunLog(os.path.join(EXPERIMENT_DIR, "logs", "run.log"))
    report = run_consistency(source_hash, log)
    write_json(os.path.join(EXPERIMENT_DIR, "benchmarks", "consistency.json"), report)
    print(json.dumps(report, indent=2))
    log.close()


if __name__ == "__main__":
    main()
