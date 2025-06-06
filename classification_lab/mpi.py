from mpi4py import MPI
import numpy as np
import time
from main import load_mnist_data
from heapq import heappush, heappushpop

# ========== KNN 函数 ==========

def euclidean_distance(x1, x2):
    """计算两个特征向量之间的欧氏距离"""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn_serial(X_train, y_train, X_test, y_test, k=5):
    """串行版本KNN分类器"""
    y_pred = []
    correct = 0
    start_time = time.time()
    
    # 遍历每个测试样本
    for i, x_test in enumerate(X_test):
        # 展平图像
        flat_test = x_test.flatten()
        neighbors = []  # 使用最大堆保存最近的k个邻居
        
        # 计算与所有训练样本的距离
        for idx, x_train in enumerate(X_train):
            # 展平训练图像
            flat_train = x_train.flatten()
            dist = euclidean_distance(flat_test, flat_train)
            
            # 使用堆高效维护k个最小距离
            if len(neighbors) < k:
                heappush(neighbors, (-dist, y_train[idx]))  # 负距离用于最大堆
            else:
                heappushpop(neighbors, (-dist, y_train[idx]))
        
        # 多数投票决定分类结果
        counts = {}
        for _, label in neighbors:
            counts[label] = counts.get(label, 0) + 1
        pred = max(counts.items(), key=lambda x: x[1])[0]
        
        y_pred.append(pred)
        if pred == y_test[i]:
            correct += 1  # 统计正确分类数
            
    accuracy = correct / len(X_test)
    return accuracy, correct, len(X_test)


# ========== 主程序 ==========
if __name__ == "__main__":
    # 初始化 MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 当前进程的编号
    size = comm.Get_size()  # 总进程数

    # 配置参数
    raw_folder = './data/MNIST/raw'  # 原始数据文件夹
    train_size = 1000              # 训练集大小
    test_size = 200                # 测试集大小
    k = 5                           # KNN 参数

    # 加载数据（假设数据已预处理为 numpy 数组）
    if rank == 0:
        # 主节点加载数据
        x_train, y_train, x_test, y_test = load_mnist_data(
            raw_folder, 
            train_size=train_size, 
            test_size=test_size
        )

        # 将测试数据分片
        chunks = np.array_split(x_test, size)
        labels_chunks = np.array_split(y_test, size)
    else:
        # 其他节点初始化空变量
        x_train = None
        y_train = None
        chunks = None
        labels_chunks = None

    # 广播训练数据到所有节点
    x_train = comm.bcast(x_train, root=0)
    y_train = comm.bcast(y_train, root=0)

    # 分发测试数据到各节点
    x_test_chunk = comm.scatter(chunks, root=0)
    y_test_chunk = comm.scatter(labels_chunks, root=0)

    # 各节点运行 KNN
    start_time = time.time()
    local_accuracy, local_corrects, local_size = knn_serial(x_train, y_train, x_test_chunk, y_test_chunk, k)
    local_time = time.time() - start_time
    print(f"节点 {rank} 完成计算，局部准确率: {local_accuracy:.4f}, 耗时: {local_time:.2f} 秒")

    # 收集所有节点的准确率
    # accuracies = comm.gather(local_accuracy, root=0)
    corrects = comm.gather(local_corrects, root=0)
    sizes = comm.gather(local_size, root=0)

    # 主节点汇总结果
    if rank == 0:
        total_accuracy = sum(corrects) / sum(sizes)
        print(f"总准确率: {total_accuracy:.4f}")