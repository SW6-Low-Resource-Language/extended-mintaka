#!/bin/bash
#SBATCH --job-name=llm_training
#SBATCH --output=llm_training.out
#SBATCH --error=llm_training.err
#SBATCH --mem=24G
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --mail-user=mlundg22@student.aau.dk
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=128 

#hostname

export CUDA_VISIBLE_DEVICES=$(echo $(seq -s, 0 $((SLURM_GPUS_ON_NODE-1))))
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Step 2: Run PyTorch-related tasks in the second container
singularity exec --nv --bind ~/my-virtual-env:/my-virtual-env /ceph/container/pytorch/pytorch_25.01.sif /bin/bash -c "
    source /my-virtual-env/bin/activate &&
    python train_llm2.py
"
