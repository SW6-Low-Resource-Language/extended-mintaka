#!/bin/bash
#SBATCH --job-name=mt5_fine-tune
#SBATCH --output=mt5_train.out
#SBATCH --error=mt5_train.err
#SBATCH --mem=128G
#SBATCH --gres=gpu:6
#SBATCH --time=12:00:00
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6 
#SBATCH --mail-user=mlundg22@student.aau.dk
#SBATCH --mail-type=ALL


export CUDA_VISIBLE_DEVICES=$(echo $(seq -s, 0 $((SLURM_GPUS_ON_NODE-1))))

singularity exec --nv --bind ~/my-virtual-env:/my-virtual-env /ceph/container/pytorch/pytorch_25.01.sif /bin/bash -c "
    source /my-virtual-env/bin/activate &&
    python -m torch.distributed.launch --nproc_per_node=6 fine-tune_mt5.py
"
