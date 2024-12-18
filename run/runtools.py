import os
import sys
import subprocess

####################################################################################################
def write_slurm_script_to_file(script_content, file_name):
    with open(file_name, 'w') as file:
        file.write(script_content)

def submit_slurm_script(file_name):
    result = subprocess.run(['sbatch', file_name], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Job submitted successfully: {result.stdout}")
    else:
        print(f"Error in job submission: {result.stderr}")

####################################################################################################

# ========== anaconda ==========
# ========== anaconda ==========
def generate_slurm_script_jax0426(gpustr, command):
    slurm_script = f"""#!/bin/bash
{gpustr}

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo The current job ID is $SLURM_JOB_ID
echo Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo Using $SLURM_NTASKS_PER_NODE tasks per node
echo A total of $SLURM_NTASKS tasks is used
echo List of CUDA devices: $CUDA_VISIBLE_DEVICES
echo

echo ==== Job started at `date` ====
echo

source /home/zhangqi/anaconda3/etc/profile.d/conda.sh
conda activate jax0426
module purge
module load slurm/slurm/19.05.8
module load cuda12.4/toolkit/12.4.0

cd /home/zhangqi/t02code/lithium
nvcc --version
python3 --version
pip show jax
nvidia-smi

{command}

echo
echo ==== Job finished at `date` ====
"""
    return slurm_script

####################################################################################################

# ========== anaconda ==========
def generate_slurm_script_jax0429(gpustr, command):
    slurm_script = f"""#!/bin/bash
{gpustr}

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo The current job ID is $SLURM_JOB_ID
echo Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo Using $SLURM_NTASKS_PER_NODE tasks per node
echo A total of $SLURM_NTASKS tasks is used
echo List of CUDA devices: $CUDA_VISIBLE_DEVICES
echo

echo ==== Job started at `date` ====
echo

source /home/zhangqi/anaconda3/etc/profile.d/conda.sh
conda activate jax0429
module purge
module load slurm/slurm/19.05.8
module load cuda12.5/toolkit/12.5.0

cd /home/zhangqi/t02code/lithium
nvcc --version
python3 --version
pip show jax
nvidia-smi

{command}

echo
echo ==== Job finished at `date` ====
"""
    return slurm_script

####################################################################################################

# ========== anaconda ==========
def generate_slurm_script_jax0431(gpustr, command):
    slurm_script = f"""#!/bin/bash
{gpustr}

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo The current job ID is $SLURM_JOB_ID
echo Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo Using $SLURM_NTASKS_PER_NODE tasks per node
echo A total of $SLURM_NTASKS tasks is used
echo List of CUDA devices: $CUDA_VISIBLE_DEVICES
echo

echo ==== Job started at `date` ====
echo

source /home/zhangqi/anaconda3/etc/profile.d/conda.sh
conda activate jax0431
module purge
module load slurm/slurm/19.05.8
module load cuda12.5/toolkit/12.5.0

cd /home/zhangqi/t02code/lithium
nvcc --version
python3 --version
pip show jax
nvidia-smi

{command}

echo
echo ==== Job finished at `date` ====
"""
    return slurm_script


####################################################################################################
#========== singularity ==========
def generate_slurm_script_singularity(gpustr, command):
    slurm_script = f"""#!/bin/bash
{gpustr}

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo The current job ID is $SLURM_JOB_ID
echo Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo Using $SLURM_NTASKS_PER_NODE tasks per node
echo A total of $SLURM_NTASKS tasks is used
echo List of CUDA devices: $CUDA_VISIBLE_DEVICES
echo

echo ==== Job started at `date` ====
echo

module purge
singularity exec --no-home --nv --bind /data:/data,/home/zhangqi/MLCodes:/home/zhangqi/MLCodes \\
    /home/zhangqi/MLCodes/images/ubuntu22.04-cuda12.6-jax0431-v2.sif bash -c \\
"
nvcc --version
which python3
python3 --version
pip show jax
nvidia-smi
cd /home/zhangqi/t02code/lithium

{command}
"

echo
echo ==== Job finished at `date` ====
"""
    return slurm_script


####################################################################################################
#========== on para server ==========
def generate_slurm_script_para(gpustr, command):
    slurm_script = f"""#!/bin/bash
{gpustr}

#### source ~/.bashrc

echo The current job ID is $SLURM_JOB_ID
echo Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo Using $SLURM_NTASKS_PER_NODE tasks per node
echo A total of $SLURM_NTASKS tasks is used
echo List of CUDA devices: $CUDA_VISIBLE_DEVICES
echo

echo ==== Job started at `date` ====
echo

module purge
singularity exec --no-home --nv --bind /data:/data \
    /data/home/scv9615/archive/images/ubuntu22.04-cuda12.6-jax0431-v2.sif bash -c \
"
nvcc --version
python3 --version
pip show jax
nvidia-smi
cd /data/home/scv9615/run/lithium

{command}
"

echo
echo ==== Job finished at `date` ====
"""
    return slurm_script