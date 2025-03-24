import subprocess
import os

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code {e.returncode}")
        return e

def main():
    # Clear PYTHONPATH and activate environment
    os.system("export PYTHONPATH=")
    os.system("conda deactivate")
    os.system("conda activate cotta")

    # Run source and norm configurations
    # run_command("CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/source.yaml")
    # run_command("CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/norm.yaml")

    # Run tent and cotta configurations in a loop
    # for i in range(10):
    tent_cmd = f"CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/10orders/tent/tent10.yaml"
    run_command(tent_cmd)

    cotta_cmd = f"CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/10orders/cotta/cotta10.yaml"
    run_command(cotta_cmd)

    # Run evaluation and save log
    os.chdir("output")
    run_command("python3 -u ../eval.py | tee result.log")

if __name__ == "__main__":
    main()