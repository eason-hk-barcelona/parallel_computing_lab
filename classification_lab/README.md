
# README

## How to Run
1. prepare the dataset(MNIST CSV)

    ``` 
    from torchvision import datasets, transforms
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    ```

2. install requirments
   ```
   pip install mrjob numpy torch torchvision
   ```

3. run
    ```
    python main.py
    ```
