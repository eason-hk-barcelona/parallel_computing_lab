#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <algorithm>
#include <immintrin.h>

const double PI = std::acos(-1.0);

// 串行位反转置换
void serial_bit_reverse(std::vector<std::complex<double>>& data) {
    int n = data.size();
    int log_n = static_cast<int>(std::log2(n));
    
    for (int i = 0; i < n; i++) {
        int j = 0;
        int temp_i = i;
        for (int k = 0; k < log_n; k++) {
            j = (j << 1) | (temp_i & 1);
            temp_i >>= 1;
        }
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
}

// 串行FFT实现
void serial_fft(std::vector<std::complex<double>>& data) {
    int n = data.size();
    serial_bit_reverse(data);
    
    for (int step = 2; step <= n; step <<= 1) {
        int half_step = step >> 1;
        std::complex<double> w_step = std::exp(std::complex<double>(0, -2.0 * PI / step));
        
        for (int i = 0; i < n; i += step) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < half_step; j++) {
                int even = i + j;
                int odd = i + j + half_step;
                
                std::complex<double> t = w * data[odd];
                std::complex<double> u = data[even];
                
                data[even] = u + t;
                data[odd] = u - t;
                
                w *= w_step;
            }
        }
    }
}


// 预计算位反转索引表
std::vector<int> precompute_bit_reverse_indices(int n) {
    int log_n = 0;
    int temp = n;
    while (temp > 1) {
        temp >>= 1;
        log_n++;
    }
    
    std::vector<int> rev_index(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int j = 0;
        int temp_i = i;
        for (int k = 0; k < log_n; k++) {
            j = (j << 1) | (temp_i & 1);
            temp_i >>= 1;
        }
        rev_index[i] = j;
    }
    return rev_index;
}

// 使用预计算索引表的位反转置换
void optimized_bit_reverse(std::vector<std::complex<double>>& data, 
                          const std::vector<int>& rev_index) {
    int n = data.size();
    for (int i = 0; i < n; i++) {
        int j = rev_index[i];
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
}

#ifdef __AVX2__
void avx2_butterfly(std::complex<double>* data, int start, int half_step, 
                    const std::complex<double>& w_step) {
    // 加载旋转因子
    __m256d w_real = _mm256_set1_pd(w_step.real());
    __m256d w_imag = _mm256_set1_pd(w_step.imag());
    
    // 当前旋转因子
    __m256d w_real_current = _mm256_set1_pd(1.0);
    __m256d w_imag_current = _mm256_set1_pd(0.0);
    
    for (int j = 0; j < half_step; j += 2) {  // 一次处理2个蝶形运算
        // 加载偶部元素 (2个复数 = 4个double)
        int even_idx = start + j;
        __m256d even = _mm256_loadu_pd(reinterpret_cast<double*>(&data[even_idx]));
        
        // 加载奇部元素
        int odd_idx = even_idx + half_step;
        __m256d odd = _mm256_loadu_pd(reinterpret_cast<double*>(&data[odd_idx]));
        
        // 计算 w * odd
        // 实部: w_real * odd_real - w_imag * odd_imag
        // 虚部: w_real * odd_imag + w_imag * odd_real
        __m256d odd_real = _mm256_permute_pd(odd, 0b0101); // 交换实虚部
        __m256d t_real = _mm256_mul_pd(w_real_current, odd);
        __m256d t_imag = _mm256_mul_pd(w_imag_current, odd_real);
        __m256d t = _mm256_addsub_pd(t_real, t_imag);
        
        // 计算 u + t 和 u - t
        __m256d u_plus_t = _mm256_add_pd(even, t);
        __m256d u_minus_t = _mm256_sub_pd(even, t);
        
        // 存储结果
        _mm256_storeu_pd(reinterpret_cast<double*>(&data[even_idx]), u_plus_t);
        _mm256_storeu_pd(reinterpret_cast<double*>(&data[odd_idx]), u_minus_t);
        
        // 更新旋转因子: w = w * w_step
        __m256d w_real_next = _mm256_sub_pd(
            _mm256_mul_pd(w_real_current, w_real),
            _mm256_mul_pd(w_imag_current, w_imag)
        );
        __m256d w_imag_next = _mm256_add_pd(
            _mm256_mul_pd(w_real_current, w_imag),
            _mm256_mul_pd(w_imag_current, w_real)
        );
        
        w_real_current = w_real_next;
        w_imag_current = w_imag_next;
    }
}
#endif



// AVX-512 向量化蝶形运算
#ifdef __AVX512F__
void avx512_butterfly(std::complex<double>* data, int start, int half_step, 
                     const std::complex<double>& w_step) {
    // 加载旋转因子
    __m512d w_real = _mm512_set1_pd(w_step.real());
    __m512d w_imag = _mm512_set1_pd(w_step.imag());
    
    // 当前旋转因子
    __m512d w_real_current = _mm512_set1_pd(1.0);
    __m512d w_imag_current = _mm512_set1_pd(0.0);
    
    for (int j = 0; j < half_step; j += 4) {  // 一次处理4个蝶形运算
        // 加载偶部元素 (4个复数 = 8个double)
        int even_idx = start + j;
        __m512d even = _mm512_loadu_pd(reinterpret_cast<double*>(&data[even_idx]));
        
        // 加载奇部元素
        int odd_idx = even_idx + half_step;
        __m512d odd = _mm512_loadu_pd(reinterpret_cast<double*>(&data[odd_idx]));
        
        // 分解奇部为实部和虚部
        // 奇数索引元素是虚部：1,3,5,7
        // 偶数索引元素是实部：0,2,4,6
        __m512d odd_real = _mm512_shuffle_pd(odd, odd, 0b00000000); // 实部
        __m512d odd_imag = _mm512_shuffle_pd(odd, odd, 0b11111111); // 虚部
        
        // 计算 t = w * odd
        // 实部: w_real * odd_real - w_imag * odd_imag
        // 虚部: w_real * odd_imag + w_imag * odd_real
        __m512d t_real = _mm512_sub_pd(
            _mm512_mul_pd(w_real_current, odd_real),
            _mm512_mul_pd(w_imag_current, odd_imag)
        );
        
        __m512d t_imag = _mm512_add_pd(
            _mm512_mul_pd(w_real_current, odd_imag),
            _mm512_mul_pd(w_imag_current, odd_real)
        );
        
        // 交错实部和虚部
        __m512d t = _mm512_unpacklo_pd(t_real, t_imag);
        __m512d t_high = _mm512_unpackhi_pd(t_real, t_imag);
        t = _mm512_shuffle_f64x2(t, t_high, 0x44);
        
        // 计算 u + t 和 u - t
        __m512d u_plus_t = _mm512_add_pd(even, t);
        __m512d u_minus_t = _mm512_sub_pd(even, t);
        
        // 存储结果
        _mm512_storeu_pd(reinterpret_cast<double*>(&data[even_idx]), u_plus_t);
        _mm512_storeu_pd(reinterpret_cast<double*>(&data[odd_idx]), u_minus_t);
        
        // 更新旋转因子: w = w * w_step
        __m512d w_real_next = _mm512_sub_pd(
            _mm512_mul_pd(w_real_current, w_real),
            _mm512_mul_pd(w_imag_current, w_imag)
        );
        
        __m512d w_imag_next = _mm512_add_pd(
            _mm512_mul_pd(w_real_current, w_imag),
            _mm512_mul_pd(w_imag_current, w_real)
        );
        
        w_real_current = w_real_next;
        w_imag_current = w_imag_next;
    }
}
#endif

// 优化后的并行FFT
void optimized_fft_openmp(std::vector<std::complex<double>>& data) {
    int n = data.size();
    
    // 小规模数据使用串行
    if (n <= 4096) {
        serial_fft(data);
        return;
    }
    
    // 预计算位反转索引表
    std::vector<int> rev_index = precompute_bit_reverse_indices(n);
    
    // 步骤1：位反转置换
    optimized_bit_reverse(data, rev_index);
    
    // 步骤2：分治式蝶形运算
    #pragma omp parallel
    {
        for (int step = 2; step <= n; step <<= 1) {
            int half_step = step >> 1;
            std::complex<double> w_step = std::exp(std::complex<double>(0, -2.0 * PI / step));
            
            #pragma omp for schedule(static)
            for (int i = 0; i < n; i += step) {
                #ifdef __AVX512F__
                if (half_step >= 4 && step <= n/16) { // 限制在适当规模使用AVX-512
                    avx512_butterfly(data.data(), i, half_step, w_step);
                    continue;
                }
                #endif

                // 使用向量化优化
                #ifdef __AVX2__
                if (half_step >= 2 && step <= n/8) { // 限制在适当规模使用AVX
                    avx2_butterfly(data.data(), i, half_step, w_step);
                    continue;
                }
                #endif
                
                // 标量版本
                std::complex<double> w(1.0, 0.0);
                for (int j = 0; j < half_step; j++) {
                    int even = i + j;
                    int odd = i + j + half_step;
                    
                    std::complex<double> t = w * data[odd];
                    std::complex<double> u = data[even];
                    
                    data[even] = u + t;
                    data[odd] = u - t;
                    
                    w *= w_step;
                }
            }
        }
    }
}


// 生成大规模测试数据 (多频复合信号)
std::vector<std::complex<double>> generate_test_signal(int n) {
    std::vector<std::complex<double>> signal(n);
    std::mt19937 gen(42); // 固定种子确保可重复性
    std::uniform_real_distribution<double> amp_dist(0.1, 1.0);
    std::uniform_real_distribution<double> phase_dist(0, 2*PI);
    
    // 生成3个主要频率成分
    double f1 = 100.0 * 2*PI/n;
    double f2 = 250.0 * 2*PI/n;
    double f3 = 500.0 * 2*PI/n;
    
    // 随机小振幅噪声频率
    std::vector<double> noise_freqs(20);
    std::vector<double> noise_amps(20);
    std::vector<double> noise_phases(20);
    for (int i = 0; i < 20; i++) {
        noise_freqs[i] = (10 + i*50) * 2*PI/n;
        noise_amps[i] = amp_dist(gen) * 0.05;
        noise_phases[i] = phase_dist(gen);
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double t = static_cast<double>(i)/n;
        
        // 主信号
        double value = 1.2 * sin(f1*i) 
                     + 0.8 * cos(f2*i + PI/4)
                     + 0.5 * sin(f3*i + PI/3);
        
        // 添加噪声
        for (int j = 0; j < 20; j++) {
            value += noise_amps[j] * sin(noise_freqs[j]*i + noise_phases[j]);
        }
        
        signal[i] = {value, 0.0};
    }
    
    return signal;
}

// 验证FFT结果正确性
bool validate_results(const std::vector<std::complex<double>>& serial_result,
                      const std::vector<std::complex<double>>& parallel_result,
                      double tolerance = 1e-6) {
    if (serial_result.size() != parallel_result.size()) {
        std::cerr << "Error: Result size mismatch\n";
        return false;
    }
    
    double max_error = 0.0;
    double max_rel_error = 0.0;
    int error_count = 0;
    
    for (size_t i = 0; i < serial_result.size(); i++) {
        double real_diff = std::abs(serial_result[i].real() - parallel_result[i].real());
        double imag_diff = std::abs(serial_result[i].imag() - parallel_result[i].imag());
        
        double magnitude = std::abs(serial_result[i]);
        double rel_error = (magnitude > 1e-12) ? 
                          (real_diff + imag_diff) / magnitude : 
                          real_diff + imag_diff;
        
        if (rel_error > tolerance) {
            if (error_count < 10) { // 只打印前10个错误
                std::cout << "Mismatch at index " << i << ": "
                          << "Serial(" << serial_result[i].real() << ", " << serial_result[i].imag() << ") "
                          << "Parallel(" << parallel_result[i].real() << ", " << parallel_result[i].imag() << ") "
                          << "Rel error: " << rel_error << "\n";
            }
            error_count++;
        }
        
        max_error = std::max(max_error, real_diff + imag_diff);
        max_rel_error = std::max(max_rel_error, rel_error);
    }
    
    if (error_count > 0) {
        std::cout << "Validation failed: " << error_count << " errors found\n";
        std::cout << "Max absolute error: " << max_error << "\n";
        std::cout << "Max relative error: " << max_rel_error << "\n";
        return false;
    }
    
    std::cout << "Validation passed! Max relative error: " << max_rel_error << "\n";
    return true;
}

// 性能测试函数
void performance_test(int min_size, int max_size, const int thread_counts[]) {
    std::ofstream outfile("fft_performance.csv");
    outfile << "Size,Threads,SerialTime(ms),ParallelTime(ms),Speedup,Efficiency\n";
    
    for (int size = min_size; size <= max_size; size *= 2) {
        std::cout << "\nTesting size: " << size << "\n";
        auto signal = generate_test_signal(size);
        std::cout << "Generated test\n";
        
        // 串行基准测试
        auto serial_signal = signal;
        auto start_serial = std::chrono::high_resolution_clock::now();
        serial_fft(serial_signal);
        auto end_serial = std::chrono::high_resolution_clock::now();
        double serial_time = std::chrono::duration<double, std::milli>(end_serial - start_serial).count();
        std::cout << "Serial: " << serial_time << " ms" << std::endl;

        // 并行测试不同线程数
        for (int i = 0; thread_counts[i] != 0; i++) {
            int threads = thread_counts[i];
            omp_set_num_threads(threads);
            
            auto parallel_signal = signal;
            auto start_parallel = std::chrono::high_resolution_clock::now();
            optimized_fft_openmp(parallel_signal);
            auto end_parallel = std::chrono::high_resolution_clock::now();
            double parallel_time = std::chrono::duration<double, std::milli>(end_parallel - start_parallel).count();
            
            // 验证结果
            std::cout << "Validating with " << threads << " threads... ";
            bool valid = validate_results(serial_signal, parallel_signal);
            
            // 计算性能指标
            double speedup = serial_time / parallel_time;
            double efficiency = (speedup / threads) * 100;
            
            // 输出结果
            std::cout << "Threads: " << threads 
                      << " | Serial: " << serial_time << " ms"
                      << " | Parallel: " << parallel_time << " ms"
                      << " | Speedup: " << speedup
                      << " | Efficiency: " << efficiency << "%\n";
            
            // 保存到CSV
            outfile << size << "," << threads << ","
                    << serial_time << "," << parallel_time << ","
                    << speedup << "," << efficiency << "\n";
        }
    }
    
    outfile.close();
    std::cout << "\nPerformance data saved to fft_performance.csv\n";
}

// 频谱分析工具
void analyze_spectrum(const std::vector<std::complex<double>>& fft_result) {
    int n = fft_result.size();
    std::vector<double> magnitudes(n/2);
    
    // 计算幅度谱
    #pragma omp parallel for
    for (int i = 0; i < n/2; i++) {
        magnitudes[i] = std::abs(fft_result[i]);
    }
    
    // 寻找主要频率成分
    std::vector<std::pair<double, int>> peaks;
    for (int i = 1; i < n/2 - 1; i++) {
        if (magnitudes[i] > magnitudes[i-1] && magnitudes[i] > magnitudes[i+1]) {
            peaks.emplace_back(magnitudes[i], i);
        }
    }
    
    // 按幅度排序
    std::sort(peaks.begin(), peaks.end(), std::greater<>());
    
    std::cout << "\nTop 5 frequency components:\n";
    std::cout << "Index\tFrequency\tMagnitude\n";
    for (int i = 0; i < std::min(5, static_cast<int>(peaks.size())); i++) {
        double freq = static_cast<double>(peaks[i].second) / n;
        std::cout << peaks[i].second << "\t" << freq << "\t\t" << peaks[i].first << "\n";
    }
}

int main() {
    // 设置测试参数
    const int MIN_SIZE = 1 << 10;    // 1024
    const int MAX_SIZE = 1 << 24;    // 1048576
    const int THREAD_COUNTS[] = {1, 2, 4, 8, 16, 32, 64, 0};  // 0表示结束
    // const int THREAD_COUNTS[] = {8, 0};  // 0表示结束
    
    // 运行性能测试
    performance_test(MIN_SIZE, MAX_SIZE, THREAD_COUNTS);
    
    // 额外分析最大规模数据的频谱
    std::cout << "\nAnalyzing spectrum for size " << MAX_SIZE << "...\n";
    auto large_signal = generate_test_signal(MAX_SIZE);
    optimized_fft_openmp(large_signal);
    analyze_spectrum(large_signal);
    
    return 0;
}