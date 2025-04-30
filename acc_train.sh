#!/bin/bash
#SBATCH --job-name=mt5_fine-tune
#SBATCH --output=acc_mt5_train.out
#SBATCH --error=acc_mt5_train.err
#SBATCH --mem=128G
#SBATCH --gres=gpu:6
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36 
#SBATCH --mail-user=mlundg22@student.aau.dk
#SBATCH --mail-type=END,FAIL


export CUDA_VISIBLE_DEVICES=$(echo $(seq -s, 0 $((SLURM_GPUS_ON_NODE-1))))
export WORLD_SIZE=6                     # Total number of GPUs
export LOCAL_RANK=0                     # Rank of the current process (set dynamically)
export RANK=0                           # Global rank (set dynamically)
export MASTER_ADDR=127.0.0.1            # Address of the master node
export MASTER_PORT=29501

singularity exec --nv --bind ~/my-virtual-env:/my-virtual-env /ceph/container/pytorch/pytorch_25.01.sif /bin/bash -c "
	source /my-virtual-env/bin/activate &&
    	python -m torch.distributed.launch --nproc_per_node=6 --master_port=29501 fine-tune.py
"
