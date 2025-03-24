import os
import subprocess

# 设置环境变量
os.environ['PYTHONPATH'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 激活 Conda 环境
conda_path = "/cluster/home/qwang/miniconda3/etc/profile.d/conda.sh"
activate_env = f'source {conda_path} && conda deactivate && conda activate cotta'
subprocess.run(activate_env, shell=True, executable='/bin/bash')


# 定义函数，运行 Python 脚本
def run_script(cfg_file, log_file):
    command = f"python -u cifar10c_gradual.py --cfg {cfg_file}"
    with open(log_file, 'w') as f:
        subprocess.run(command, shell=True, stdout=f, stderr=f)


# 运行 tent 和 cotta 的不同配置文件
for i in range(10):
    # tent 配置
    tent_cfg = f"cfgs/10orders/tent/tent{i}.yaml"
    tent_log = f"tent_{i}.log"
    run_script(tent_cfg, tent_log)

    # cotta 配置
    cotta_cfg = f"cfgs/10orders/cotta/cotta{i}.yaml"
    cotta_log = f"cotta_{i}.log"
    run_script(cotta_cfg, cotta_log)