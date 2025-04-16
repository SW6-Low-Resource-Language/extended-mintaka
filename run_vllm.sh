#!/bin/bash
#SBATCH --job-name=mt5_inference
#SBATCH --output=vllm_textgen_output_mt5_da.log
#SBATCH --error=vllm_textgen_error_mt5_da.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00

singularity exec --nv /ceph/container/vllm-openai_latest.sif python3 inference.py
