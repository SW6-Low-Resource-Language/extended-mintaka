#!/bin/bash

#SBATCH --job-name=semantic_similarity_da
#SBATCH --output=semantic_similarity_output.txt
#SBATCH --error=semantic_similarity_error.txt
#SBATCH --mem=24g
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10

#hostname
singularity exec --bind ~/my-virtual-env:/my-virtual-env /ceph/container/python/python_3.12.sif /bin/bash -c "source /my-virtual-env/bin/activate && python semantic_similarity.py"
#singularity exec /ceph/container/pytorch/pytorch_25.01.sif python3 semantic_similarity.py
