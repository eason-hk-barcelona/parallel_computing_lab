
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


*MPI NOTIFICATION*

We use the `intel-oneapi-mpi@2021.13.1` as MPI-Library which will be link by mpi4py

