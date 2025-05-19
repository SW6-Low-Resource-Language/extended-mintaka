#!/bin/bash
#SBATCH --job-name=zero_shot_qwen
#SBATCH --output=qwen_32B_da.out
#SBATCH --error=qwen_32B_da.err
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=12:00:00


srun singularity exec --nv --bind ~/my-virtual-env:/my-virtual-env /ceph/container/pytorch/pytorch_25.01.sif /bin/bash -c "source /my-virtual-env/bin/activate && python3 qwen_inf2.py"
#singularity exec --nv /ceph/container/vllm-openai_latest.sif python3 qwen_inf1.py
