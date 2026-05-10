#!/usr/bin/env python3
"""Run all τ-injection experiments to reproduce paper results.
Usage: D:\python\python.exe rwkv/experiments/run_all.py [--model 0.4B|1.5B|2.9B|all]"""
import sys, os, argparse

TAU_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TAU_DIR)
sys.path.insert(0, os.path.join(TAU_DIR, "deploy_pkg"))

from tau_injection import get_device
DEV = get_device()
print(f"Device: {'GPU: ' + torch.cuda.get_device_name(0) if DEV == 'cuda' else 'CPU'}" if 'torch' in dir() else f"Device: {DEV}")

import torch

MODELS = {
    "0.4B": r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
    "1.5B": r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth",
    "2.9B": r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-2.9B-v2.1-20250122-ctx4096.pth",
}

def run_all(model_filter="all"):
    models_to_run = {k: v for k, v in MODELS.items() if model_filter == "all" or k == model_filter}

    for name, path in models_to_run.items():
        print(f"\n{'='*70}")
        print(f"RWKV-7-{name}")
        print(f"{'='*70}")

        from rwkv.experiments import experiment_01_grid_search
        experiment_01_grid_search.run(name, path)

        from rwkv.experiments import experiment_02_v_injection
        experiment_02_v_injection.run(name, path)

        from rwkv.experiments import experiment_03_sweep
        experiment_03_sweep.run(name, path)

        from rwkv.experiments import experiment_04_dynamics
        experiment_04_dynamics.run(name, path)

        if name != "2.9B":
            from rwkv.experiments import experiment_05_generation
            experiment_05_generation.run(name, path)

    print("\nAll experiments complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=["0.4B", "1.5B", "2.9B", "all"])
    args = parser.parse_args()
    run_all(args.model)