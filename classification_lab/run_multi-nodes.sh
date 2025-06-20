#!/bin/bash
#SBATCH --job-name=knn_parallel
#SBATCH --output=knn_parallel-%j.out
#SBATCH --error=knn_parallel-%j.err
#SBATCH --nodes=4                # 使用 4 个节点
#SBATCH --ntasks-per-node=2      # 每个节点运行 1 个任务
#SBATCH --cpus-per-task=32        # 每个任务使用 4 个 CPU


# 运行程序
srun --mpi=pmix python mpi.py --train_size 10000 --test_size 2000 \
--k 5 --num_workers 32