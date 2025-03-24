
import subprocess
import os
def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code {e.returncode}")
        return e
run_command("CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/source.yaml")