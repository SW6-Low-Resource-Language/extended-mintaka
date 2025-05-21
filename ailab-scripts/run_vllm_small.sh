#!/bin/bash
#SBATCH --job-name=vllm_textgen
#SBATCH --output=vllm_textgen_output.log
#SBATCH --error=vllm_textgen_error.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00

singularity exec --nv /ceph/container/vllm-openai_latest.sif python3 1b_inference.py
