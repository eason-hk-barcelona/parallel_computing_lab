
# README

## How to Run
1. prepare the dataset(MNIST CSV)

    ``` 
    from torchvision import datasets, transforms
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    ```

2. install requirments
   ```
   pip install mrjob numpy torch torchvision mpi4py
   ```

3. run
    ```
    python main.py
    ```

    *multi-nodes*

    We use the *slurm* to manage tasks.
    ```
    sbatch run_multi-nodes.sh     
    ```
    **Addition** :
    
    In run_multi-nodes.sh, you can change the args to fix the MPI_nodes and proc per MPI_task.
    
    `--num_workers`: num of proc per MPI_task;
    `--cpus-per-task`: cpus per slurm task: MPI_task;
    `--nodes`: num of nodes
    `--ntasks-per-node`: whats means like its name;

    Total_MPI_tasks = nodes * ntasks-per-nodes;


*MPI NOTIFICATION*

We use the `intel-oneapi-mpi@2021.13.1` as MPI-Library which will be link by mpi4py

