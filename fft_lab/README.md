# 2D Fast Fourier Transform (FFT) Implementation

这个项目实现了2D快速傅里叶变换(FFT)的串行版本和混合并行版本(MPI+OpenMP)。项目能够处理PGM格式的图像文件，执行正向和逆向FFT变换，并输出结果图像。

## 项目结构

```
fft_lab/
├── fft_serial.c      # 串行FFT实现
├── fft_hybrid.c      # 混合并行FFT实现 (MPI+OpenMP)
├── pgm.c            # PGM图像I/O操作
├── pgm.h            # PGM图像结构定义
├── cshift.c         # FFT频域移位函数
├── cshift.h         # 移位函数头文件
├── Makefile         # 编译配置文件
├── test_fft.sh      # 自动化测试脚本
├── README.md        # 项目说明文档
├── Baboon_256.pgm   # 测试图像 (256x256)
├── cube_1024.pgm    # 测试图像 (1024x1024)
└── imgs/            # 输出图像目录
```

## 功能特性

### 串行版本 (fft_serial.c)
- **算法**: Cooley-Tukey FFT算法
- **特性**: 
  - 自动零填充到2的幂次方
  - 正向和逆向FFT变换
  - FFT频域移位用于可视化
  - 支持任意尺寸的方形图像

### 混合并行版本 (fft_hybrid.c)
- **并行模型**: MPI + OpenMP混合并行
- **特性**:
  - MPI进程间分布式计算
  - OpenMP线程级并行优化
  - 动态负载均衡
  - 高效的矩阵转置和数据重分布

## 编译和构建

### 前置要求
- GCC编译器 (支持C99标准)
- OpenMPI库
- OpenMP支持

### 编译选项

```bash
# 编译所有版本
make all

# 仅编译串行版本
make fft_serial

# 仅编译混合并行版本
make fft_hybrid

# 清理所有构建文件
make clean

# 仅清理目标文件
make clean-objects

# 仅清理输出图像
make clean-output

# 整理现有的PGM文件到imgs目录
make organize-imgs

# 显示帮助信息
make help
```

### 编译标志说明
- **串行版本**: `-Wall -O2 -lm`
- **混合版本**: `-Wall -O2 -lm -fopenmp` (MPI编译器)

## 使用方法

### 1. 直接运行

#### 串行版本
```bash
./fft_serial <input_image.pgm>
```

#### 混合并行版本
```bash
# 设置OpenMP线程数
export OMP_NUM_THREADS=4

# 运行MPI程序
mpirun -np 2 ./fft_hybrid <input_image.pgm>
```

### 2. 使用测试脚本

`test_fft.sh` 脚本提供了自动化的测试和性能分析功能：

```bash
# 给脚本添加执行权限
chmod +x test_fft.sh

# 运行所有测试
./test_fft.sh all

# 仅测试串行版本
./test_fft.sh serial

# 仅测试混合并行版本
./test_fft.sh hybrid

# 仅运行性能对比测试
./test_fft.sh performance
```

### 3. 测试配置

脚本会自动测试以下混合并行配置 (总共12个CPU核心)：
- 1进程 × 12线程
- 2进程 × 6线程
- 3进程 × 4线程
- 4进程 × 3线程
- 6进程 × 2线程
- 12进程 × 1线程

## 输出文件

程序会生成以下输出文件：
- `fft.pgm`: FFT变换后的频域图像 (对数缩放)
- `ifft.pgm`: 逆FFT变换后的恢复图像

测试脚本会将输出文件自动移动到 `imgs/` 目录，并按以下格式命名：
- `fft_serial_<image_name>.pgm`
- `ifft_serial_<image_name>.pgm`
- `fft_hybrid_<image_name>_p<processes>t<threads>.pgm`
- `ifft_hybrid_<image_name>_p<processes>t<threads>.pgm`

## 性能分析

### 性能指标
- **Pure FFT Time**: 纯FFT计算时间
- **Pure IFFT Time**: 纯IFFT计算时间 (仅串行版本)
- **Real Time**: 总执行时间 (包括I/O)
- **User Time**: 用户空间CPU时间
- **Sys Time**: 系统空间CPU时间


## 算法说明

### 2D FFT算法流程
1. **预处理**: 零填充至2的幂次方和方形图像
2. **行FFT**: 对每一行执行1D FFT
3. **矩阵转置**: 将行转换为列来处理
4. **列FFT**: 对每一列执行1D FFT
5. **后处理**: FFT移位和格式转换

### 并行策略
- **MPI**: 按行分布数据到不同进程
- **OpenMP**: 在每个进程内并行处理多行
- **通信**: 使用MPI_Scatterv/Gatherv进行数据分发和收集

## 故障排除

### 常见问题

1. **编译错误**
   ```bash
   # 检查MPI是否安装
   which mpicc
   
   # 检查OpenMP支持
   gcc -fopenmp --version
   ```

2. **运行时错误**
   ```bash
   # 检查输入文件是否存在
   ls -la *.pgm
   
   # 检查MPI环境
   mpirun --version
   ```

3. **性能问题**
   - 调整MPI进程数和OpenMP线程数的组合
   - 检查系统负载和内存使用情况
   - 确保进程数不超过可用CPU核心数

