#!/bin/bash
#SBATCH --job-name=mt5_inference
#SBATCH --output=textgen_mt5_da.out
#SBATCH --error=textgen_mt5_da.err
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00

singularity exec --nv -B ~/my_venv /ceph/container/pytorch/pytorch_25.01.sif /bin/bash -c "
	source ~/my_venv/bin/activate &&
	python3 inference_mt5_checkpoint.py
"
