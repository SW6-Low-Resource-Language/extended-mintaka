#!/bin/bash
#SBATCH --job-name=zero_shot_llama
#SBATCH --output=vllm_textgen_output_llama_fi.log
#SBATCH --error=vllm_textgen_error_llama_fi.log
#SBATCH --gres=gpu:8
#SBATCH --mem=192G
#SBATCH --time=12:00:00

singularity exec --nv /ceph/container/vllm-openai_latest.sif python3 llama_inference.py
