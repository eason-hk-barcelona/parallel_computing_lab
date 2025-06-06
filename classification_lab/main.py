import numpy as np
import time
import os
import struct
from heapq import heappush, heappushpop
from mrjob.job import MRJob
from mrjob.step import MRStep


def load_idx_images(file_path):
    """加载IDX格式的图像文件"""
    with open(file_path, 'rb') as f:
        # 读取文件头部信息
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2051:
            raise ValueError(f"Invalid magic number {magic_number} in image file: {file_path}")
        
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        
        # 读取图像数据
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        return image_data.reshape(num_images, rows, cols) / 255.0  # 归一化到 [0, 1]

def load_idx_labels(file_path):
    """加载IDX格式的标签文件"""
    with open(file_path, 'rb') as f:
        # 读取文件头部信息
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2049:
            raise ValueError(f"Invalid magic number {magic_number} in label file: {file_path}")
        
        num_labels = struct.unpack('>I', f.read(4))[0]
        
        # 读取标签数据
        label_data = np.frombuffer(f.read(), dtype=np.uint8)
        return label_data

def load_mnist_data(raw_folder, train_size=None, test_size=None):
    """
    从原始文件夹加载MNIST数据集
    :param raw_folder: 包含MNIST原始文件的文件夹路径
    :param train_size: 训练集大小限制 (可选)
    :param test_size: 测试集大小限制 (可选)
    :return: (训练图像, 训练标签, 测试图像, 测试标签)
    """
    # 构建文件路径
    train_images_path = os.path.join(raw_folder, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(raw_folder, 'train-labels-idx1-ubyte')
    test_images_path = os.path.join(raw_folder, 't10k-images-idx3-ubyte')
    test_labels_path = os.path.join(raw_folder, 't10k-labels-idx1-ubyte')
    
    # 加载数据
    x_train = load_idx_images(train_images_path)
    y_train = load_idx_labels(train_labels_path)
    x_test = load_idx_images(test_images_path)
    y_test = load_idx_labels(test_labels_path)
    
    # 应用大小限制
    if train_size:
        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
    
    if test_size:
        x_test = x_test[:test_size]
        y_test = y_test[:test_size]
    
    return x_train, y_train, x_test, y_test

def save_to_csv(images, labels, output_path):
    """将图像和标签保存为CSV格式，供MapReduce使用"""
    with open(output_path, 'w') as f:
        for i in range(len(images)):
            # 展平图像并转换为字符串
            flattened = images[i].flatten()
            line = [str(labels[i])] + [str(pixel) for pixel in flattened]
            f.write(','.join(line) + '\n')

# 欧氏距离计算
def euclidean_distance(x1, x2):
    """计算两个特征向量之间的欧氏距离"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

# 串行KNN实现
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
    return accuracy, time.time() - start_time

# MapReduce KNN实现
class MRKNN(MRJob):
    """基于MapReduce的并行KNN分类器"""
    
    def configure_args(self):
        super(MRKNN, self).configure_args()
        self.add_file_arg('--train')  # 训练集文件路径
        self.add_passthru_arg('--k', type=int, default=5)  # K值参数
    
    def mapper_init(self):
        """Mapper初始化：加载训练数据集"""
        self.X_train, self.y_train = self.load_training_data()
        self.k = self.options.k
    
    def load_training_data(self):
        """解析训练集CSV文件"""
        # 加载训练数据
        train_data = np.loadtxt(self.options.train, delimiter=',')
        
        # 第一列为标签，其余为像素数据
        y_train = train_data[:, 0].astype(int)
        X_train = train_data[:, 1:]
        
        return X_train, y_train
    
    def mapper(self, _, line):
        """Map函数：处理单个测试样本"""
        # 解析CSV行
        data = np.array([float(x) for x in line.split(',')])
        true_label = int(data[0])
        x_test = data[1:]  # 像素数据不需要再归一化，因为保存时已经归一化
        
        neighbors = []
        # 计算与所有训练样本的距离
        for i in range(len(self.X_train)):
            dist = euclidean_distance(x_test, self.X_train[i])
            
            # 使用堆维护最近的k个邻居
            if len(neighbors) < self.k:
                heappush(neighbors, (-dist, self.y_train[i]))
            else:
                heappushpop(neighbors, (-dist, self.y_train[i]))
        
        # 多数投票预测标签
        counts = {}
        for _, label in neighbors:
            counts[label] = counts.get(label, 0) + 1
        pred = max(counts.items(), key=lambda x: x[1])[0]
        
        # 输出预测是否正确 (1=正确, 0=错误)
        yield None, (1 if pred == true_label else 0)
    
    def reducer(self, _, values):
        """Reduce函数：计算整体准确率"""
        total_correct = 0
        count = 0
        for v in values:
            total_correct += v
            count += 1
        yield "Accuracy", total_correct / count
    
    def steps(self):
        """定义MapReduce步骤"""
        return [
            MRStep(mapper_init=self.mapper_init,
                   mapper=self.mapper,
                   reducer=self.reducer)
        ]


import multiprocessing

def knn_parallel(x_train, y_train, x_test, y_test, k, num_workers=4):
    """
    使用多进程模拟并行 KNN
    """
    def worker(test_subset, results, idx):
        """
        每个进程处理一部分测试数据
        """
        acc, _ = knn_serial(x_train, y_train, test_subset, y_test[idx:idx+len(test_subset)], k)
        results[idx] = acc

    # 将测试数据分块
    chunk_size = len(x_test) // num_workers
    chunks = [x_test[i:i+chunk_size] for i in range(0, len(x_test), chunk_size)]

    # 创建共享字典存储结果
    manager = multiprocessing.Manager()
    results = manager.dict()

    # 启动多进程
    processes = []
    for i, chunk in enumerate(chunks):
        p = multiprocessing.Process(target=worker, args=(chunk, results, i * chunk_size))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 汇总结果
    total_acc = sum(results.values()) / len(results)
    return total_acc



if __name__ == "__main__":
    # ========== 配置参数 ==========
    raw_folder = './data/MNIST/raw'          # 原始数据文件夹
    train_size = 1000           # 训练集大小
    test_size = 200            # 测试集大小
    k = 5                       # KNN参数
    
    # 输出文件路径
    train_csv = 'mnist_train.csv'
    test_csv = 'mnist_test.csv'
    
    # ========== 加载原始数据 ==========
    print("加载原始MNIST数据集...")
    x_train, y_train, x_test, y_test = load_mnist_data(
        raw_folder, 
        train_size=train_size, 
        test_size=test_size
    )
    print(f"训练集: {len(x_train)} 样本, 测试集: {len(x_test)} 样本")
    
    # ========== 保存为CSV供MapReduce使用 ==========
    print("\n保存训练集为CSV格式...")
    save_to_csv(x_train, y_train, train_csv)
    
    print("保存测试集为CSV格式...")
    save_to_csv(x_test, y_test, test_csv)
    
    # ========== 串行KNN ==========
    print("\n运行串行KNN...")
    serial_acc, serial_time = knn_serial(x_train, y_train, x_test, y_test, k)
    print(f"串行KNN准确率: {serial_acc:.4f}, 耗时: {serial_time:.2f}秒")
    
    # ========== 并行KNN ==========
    print("\n运行并行KNN...")
    parallel_start = time.time()
    parallel_acc = knn_parallel(x_train, y_train, x_test, y_test, k, num_workers=16)
    parallel_time = time.time() - parallel_start
    print(f"并行KNN准确率: {parallel_acc:.4f}, 耗时: {parallel_time:.2f}秒")
    serial_time = serial_time if serial_time > 0 else 1  # 防止除以零
    print(f"并行KNN加速比: {serial_time / parallel_time:.2f}x")
    
    # ========== MapReduce KNN ==========
    # print("\n运行MapReduce KNN...")
    # mr_start = time.time()
    
    # # 构建MRJob参数
    # args = [
    #     'mr_knn.py',        # 脚本文件名
    #     '--train', train_csv,
    #     '--k', str(k),
    #     test_csv
    # ]
    
    # # 运行MapReduce作业
    # MRKNN(args=args).run()
    # mr_time = time.time() - mr_start
    # print(f"MapReduce总耗时: {mr_time:.2f}秒")
    
    # ========== 性能对比 ==========
    # if serial_time > 0:
    #     speedup = serial_time / mr_time
    #     print(f"\n加速比: {speedup:.2f}x")
    # else:
    #     print("\n加速比计算需要串行时间 > 0")
    
    # ========== 清理临时文件 ==========
    # 如果需要保留文件，可以注释掉以下两行
    os.remove(train_csv)
    os.remove(test_csv)
    
    print("\n实验完成!")