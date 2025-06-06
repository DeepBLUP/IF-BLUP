import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator
import mmap
import os
import time
import gc
import pandas as pd
from numba import njit, jit
import torch
from pysnptools.snpreader import Bed
import psutil
# import matplotlib.pyplot as plt
from datetime import datetime
from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator
import psutil
import os
from tqdm import tqdm
import pickle
import traceback

class EnhancedResourceMonitor:
    """增强版资源监控器"""

    def __init__(self):
        self.cpu_percent = []
        self.memory_used = []  # 实际使用的物理内存（GB）
        self.memory_percent = []
        self.timestamps = []
        self.start_time = time.time()
        self.process = psutil.Process()
        # 获取系统信息
        self.cpu_count = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB

    def update(self, step_name=""):
        # CPU使用率（所有核心的平均值）
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        self.cpu_percent.append(sum(cpu_percent) / len(cpu_percent))

        # 内存使用情况
        memory_info = self.process.memory_info()
        self.memory_used.append(memory_info.rss / (1024 ** 3))  # 转换为GB
        self.memory_percent.append(memory_info.rss / psutil.virtual_memory().total * 100)

        self.timestamps.append(time.time() - self.start_time)

        print(f"\n当前步骤: {step_name}")
        print(f"资源使用情况:")
        print(f"CPU使用率: {self.cpu_percent[-1]:.1f}% (总核心数: {self.cpu_count})")
        print(f"内存使用: {self.memory_used[-1]:.2f}GB / {self.total_memory:.2f}GB")
        print(f"内存使用率: {self.memory_percent[-1]:.1f}%")

        # 系统内存详情
        mem = psutil.virtual_memory()
        print(f"系统内存详情:")
        print(f"- 总物理内存: {mem.total / 1024 / 1024 / 1024:.2f}GB")
        print(f"- 可用内存: {mem.available / 1024 / 1024 / 1024:.2f}GB")
        print(f"- 内存使用率: {mem.percent}%")
        print(f"- 缓存使用: {mem.cached / 1024 / 1024 / 1024:.2f}GB")
        print(f"- 缓冲区使用: {mem.buffers / 1024 / 1024 / 1024:.2f}GB")

        # CPU详情
        cpu_times = psutil.cpu_times_percent()
        print(f"CPU详细使用情况:")
        print(f"- 用户空间使用率: {cpu_times.user}%")
        print(f"- 系统空间使用率: {cpu_times.system}%")
        print(f"- 空闲率: {cpu_times.idle}%")
        if hasattr(cpu_times, 'iowait'):
            print(f"- IO等待率: {cpu_times.iowait}%")

    # def plot(self, save_path='resource_usage.png'):
    #     """绘制资源使用情况图表"""
    #     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    #
    #     # CPU使用率
    #     ax1.plot(self.timestamps, self.cpu_percent)
    #     ax1.set_title(f'CPU Usage Over Time (Total Cores: {self.cpu_count})')
    #     ax1.set_ylabel('CPU %')
    #     ax1.grid(True)
    #
    #     # 内存使用量（GB）
    #     ax2.plot(self.timestamps, self.memory_used)
    #     ax2.set_title(f'Memory Usage Over Time (Total: {self.total_memory:.1f}GB)')
    #     ax2.set_ylabel('Memory Used (GB)')
    #     ax2.grid(True)
    #
    #     # 内存使用百分比
    #     ax3.plot(self.timestamps, self.memory_percent)
    #     ax3.set_title('Memory Usage Percentage')
    #     ax3.set_xlabel('Time (seconds)')
    #     ax3.set_ylabel('Memory %')
    #     ax3.grid(True)
    #
    #     plt.tight_layout()
    #     plt.savefig(save_path)
    #     plt.close()


def monitor_matrix_stats(matrix, name="矩阵"):
    """监控矩阵的基本统计信息"""
    if sp.issparse(matrix):
        nnz = matrix.nnz
        total = matrix.shape[0] * matrix.shape[1]
        sparsity = 1.0 - nnz / total
        memory_mb = (matrix.data.nbytes + matrix.indptr.nbytes +
                     matrix.indices.nbytes) / (1024 * 1024)

        print(f"\n{name}统计信息:")
        print(f"形状: {matrix.shape}")
        print(f"非零元素: {nnz:,}")
        print(f"稀疏度: {sparsity:.2%}")
        print(f"估计内存使用: {memory_mb:.2f} MB")
    else:
        memory_mb = matrix.nbytes / (1024 * 1024)
        print(f"\n{name}统计信息:")
        print(f"形状: {matrix.shape}")
        print(f"估计内存使用: {memory_mb:.2f} MB")


def get_optimal_threads(matrix_size, max_threads=32):
    """根据矩阵大小动态确定最优线程数"""
    if matrix_size < 1000:
        return min(4, max_threads)
    elif matrix_size < 5000:
        return min(8, max_threads)
    elif matrix_size < 10000:
        return min(16, max_threads)
    else:
        return min(16, max_threads)  # 避免过多线程带来的开销


class ConvergenceMonitor:
    """收敛监控器"""

    def __init__(self):
        self.residuals = []
        self.timestamps = []
        self.start_time = time.time()

    def update(self, residual):
        self.residuals.append(residual)
        self.timestamps.append(time.time() - self.start_time)

    # def plot(self, save_path='convergence.png'):
    #     plt.figure(figsize=(10, 6))
    #     plt.semilogy(self.timestamps, self.residuals)
    #     plt.title('Convergence History')
    #     plt.xlabel('Time (seconds)')
    #     plt.ylabel('Residual (log scale)')
    #     plt.grid(True)
    #     plt.savefig(save_path)
    #     plt.close()


class JacobiPreconditioner(LinearOperator):
    def __init__(self, A_UL, A_BR):
        self.dtype = np.float64
        self.n1 = A_UL.shape[0]
        self.n2 = A_BR.shape[0]
        self.shape = (self.n1 + self.n2, self.n1 + self.n2)

        try:
            if sp.issparse(A_UL):
                self.diag_UL = A_UL.diagonal().astype(self.dtype)
            else:
                self.diag_UL = np.abs(A_UL.diagonal()).astype(self.dtype)

            if sp.issparse(A_BR):
                self.diag_BR = A_BR.diagonal().astype(self.dtype)
            else:
                self.diag_BR = np.abs(A_BR.diagonal()).astype(self.dtype)
        except:
            self.diag_UL = np.ones(self.n1, dtype=self.dtype)
            self.diag_BR = np.ones(self.n2, dtype=self.dtype)

        eps = np.finfo(self.dtype).eps
        self.diag_UL = 1.0 / (self.diag_UL + eps)
        self.diag_BR = 1.0 / (self.diag_BR + eps)

    def _matvec(self, x):
        """实现矩阵向量乘法"""
        x = np.asarray(x, dtype=self.dtype)
        if x.shape != (self.shape[1],):
            raise ValueError(f"x shape mismatch. Expected {(self.shape[1],)}, got {x.shape}")

        result = np.empty(self.shape[0], dtype=self.dtype)
        result[:self.n1] = x[:self.n1] * self.diag_UL
        result[self.n1:] = x[self.n1:] * self.diag_BR
        return result

    def _rmatvec(self, x):
        """实现转置矩阵向量乘法"""
        return self._matvec(x)



class BlockDiagonalPreconditioner(LinearOperator):
    def __init__(self, A_UL, A_BR):
        self.dtype = np.float64
        self.n1 = A_UL.shape[0]
        self.n2 = A_BR.shape[0]
        self.shape = (self.n1 + self.n2, self.n1 + self.n2)

        # 只处理对角块
        if isinstance(A_UL, LinearOperator):
            self.diag_UL = np.array([A_UL.matvec(ei)[i]
                                     for i, ei in enumerate(np.eye(self.n1))])
        else:
            self.diag_UL = A_UL.diagonal() if sp.issparse(A_UL) else np.diag(A_UL)

        if isinstance(A_BR, LinearOperator):
            self.diag_BR = np.array([A_BR.matvec(ei)[i]
                                     for i, ei in enumerate(np.eye(self.n2))])
        else:
            self.diag_BR = A_BR.diagonal() if sp.issparse(A_BR) else np.diag(A_BR)

        # 平滑处理
        eps = np.finfo(self.dtype).eps
        self.diag_UL = 1.0 / (np.abs(self.diag_UL) + eps)
        self.diag_BR = 1.0 / (np.abs(self.diag_BR) + eps)

        # 添加缩放因子
        scale_UL = np.sqrt(np.mean(np.abs(self.diag_UL)))
        scale_BR = np.sqrt(np.mean(np.abs(self.diag_BR)))
        self.diag_UL *= scale_UL
        self.diag_BR *= scale_BR

    def _matvec(self, x):
        x = np.asarray(x).ravel()
        x1 = x[:self.n1]
        x2 = x[self.n1:]
        return np.concatenate([x1 * self.diag_UL, x2 * self.diag_BR])

    def _rmatvec(self, x):
        return self._matvec(x)


class SSORPreconditioner(LinearOperator):
    def __init__(self, A_UL, A_BR, omega=1.0):
        self.dtype = np.float64
        self.n1 = A_UL.shape[0]
        self.n2 = A_BR.shape[0]
        self.shape = (self.n1 + self.n2, self.n1 + self.n2)
        self.omega = omega

        # 检查矩阵大小
        print(f"Matrix sizes - A_UL: {A_UL.shape}, A_BR: {A_BR.shape}")

        # 获取对角线
        try:
            if sp.issparse(A_UL) or isinstance(A_UL, LinearOperator):
                diag_UL = np.array([A_UL.matvec(ei)[i]
                                    for i, ei in enumerate(np.eye(self.n1))])
            else:
                diag_UL = np.diag(A_UL)

            if sp.issparse(A_BR) or isinstance(A_BR, LinearOperator):
                diag_BR = np.array([A_BR.matvec(ei)[i]
                                    for i, ei in enumerate(np.eye(self.n2))])
            else:
                diag_BR = np.diag(A_BR)

            # 预计算对角线的逆
            eps = np.finfo(self.dtype).eps
            self.inv_diag_UL = 1.0 / (np.abs(diag_UL) + eps)
            self.inv_diag_BR = 1.0 / (np.abs(diag_BR) + eps)

            print("Successfully extracted diagonals")

        except Exception as e:
            print(f"Error extracting diagonals: {e}")
            # 使用单位矩阵作为后备选项
            self.inv_diag_UL = np.ones(self.n1, dtype=self.dtype)
            self.inv_diag_BR = np.ones(self.n2, dtype=self.dtype)

    def _matvec(self, x):
        """简化版本的SSOR预处理"""
        x = np.asarray(x).ravel()
        if x.shape[0] != self.shape[1]:
            raise ValueError(f"x shape mismatch. Expected {self.shape[1]}, got {x.shape[0]}")

        x1 = x[:self.n1]
        x2 = x[self.n1:]

        # 对角线缩放
        y1 = self.omega * (x1 * self.inv_diag_UL)
        y2 = self.omega * (x2 * self.inv_diag_BR)

        return np.concatenate([y1, y2])

    def _rmatvec(self, x):
        return self._matvec(x)


class ImprovedBlockPreconditioner(LinearOperator):
    def __init__(self, A_UL, A_BR, omega=0.5):
        self.dtype = np.float64
        self.n1 = A_UL.shape[0]
        self.n2 = A_BR.shape[0]
        self.shape = (self.n1 + self.n2, self.n1 + self.n2)

        print(f"Matrix sizes - A_UL: {A_UL.shape}, A_BR: {A_BR.shape}")

        # 设置松弛因子
        self.omega = omega
        print(f"Using relaxation factor: {self.omega}")

        try:
            # 针对小矩阵块（A_UL）使用直接求解
            if isinstance(A_UL, LinearOperator):
                self.diag_UL = np.array([A_UL.matvec(ei)[i]
                                         for i, ei in enumerate(np.eye(self.n1))])
                print("A_UL: Using LinearOperator")
            else:
                self.diag_UL = np.diag(A_UL)
                print("A_UL: Using direct diagonal")

            # 针对大矩阵块（A_BR）使用改进的对角线预处理
            if isinstance(A_BR, LinearOperator):
                print("A_BR: Using LinearOperator with sampling")
                # 采样策略计算对角线元素
                num_samples = min(1000, self.n2)
                sample_indices = np.linspace(0, self.n2 - 1, num_samples, dtype=int)
                self.diag_BR = np.zeros(self.n2)

                for i in sample_indices:
                    ei = np.zeros(self.n2)
                    ei[i] = 1.0
                    self.diag_BR[i] = A_BR.matvec(ei)[i]

                # 插值填充其余元素
                mask = np.zeros(self.n2, dtype=bool)
                mask[sample_indices] = True
                self.diag_BR[~mask] = np.mean(self.diag_BR[mask])
            else:
                print("A_BR: Using direct diagonal")
                self.diag_BR = np.diag(A_BR)

            # 添加自适应缩放
            eps = np.finfo(self.dtype).eps
            scale_UL = np.sqrt(np.mean(np.abs(self.diag_UL)))
            scale_BR = np.sqrt(np.mean(np.abs(self.diag_BR)))
            self.inv_diag_UL = scale_UL / (np.abs(self.diag_UL) + eps)
            self.inv_diag_BR = scale_BR / (np.abs(self.diag_BR) + eps)

            print("Successfully initialized preconditioner")

        except Exception as e:
            print(f"Error in preconditioner initialization: {e}")
            # 使用安全的后备选项
            self.inv_diag_UL = np.ones(self.n1, dtype=self.dtype)
            self.inv_diag_BR = np.ones(self.n2, dtype=self.dtype)

    def _matvec(self, x):
        x = np.asarray(x).ravel()
        if x.shape[0] != self.shape[1]:
            raise ValueError(f"x shape mismatch. Expected {self.shape[1]}, got {x.shape[0]}")

        x1 = x[:self.n1]
        x2 = x[self.n1:]

        # 应用松弛因子
        result_UL = x1 * self.inv_diag_UL
        result_BR = x2 * self.inv_diag_BR

        return np.concatenate([
            self.omega * result_UL + (1 - self.omega) * x1,  # 松弛处理 A_UL 部分
            self.omega * result_BR + (1 - self.omega) * x2  # 松弛处理 A_BR 部分
        ])

    def _rmatvec(self, x):
        return self._matvec(x)


class LargeScaleCSVReader:
    def __init__(self, filename, shape, dtype=float):
        self.filename = filename
        self.shape = shape
        self.dtype = dtype

    def __getitem__(self, key):
        with open(self.filename, 'r') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            if isinstance(key, tuple):
                row, col = key
                if isinstance(row, slice):
                    start = row.start or 0
                    stop = row.stop or self.shape[0]
                    data = []
                    mm.seek(0)
                    for _ in range(start):
                        mm.readline()
                    for _ in range(stop - start):
                        line = mm.readline().decode().strip().split(',')
                        data.append([self.dtype(x) for x in line[col]])
                    return np.array(data)
                else:
                    mm.seek(0)
                    for _ in range(row):
                        mm.readline()
                    line = mm.readline().decode().strip().split(',')
                    return self.dtype(line[col])
            mm.close()

    def get_column(self, col):
        with open(self.filename, 'r') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            data = []
            mm.seek(0)
            for _ in range(self.shape[0]):
                line = mm.readline().decode().strip().split(',')
                data.append(self.dtype(line[col]))
            mm.close()
        return np.array(data)



def building_G_matrix(bed_file, memmap_path='memmap_file', Z_memmap_path='Z_memmap_file',
                      G_memmap_path='G_memmap_file'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    time1 = time.time()

    M = Bed(f'{bed_file}', count_A1=False).read()
    num_size = M.val.shape
    M_val_memmap = np.memmap(memmap_path, dtype='float32', mode='w+', shape=M.val.shape)
    M_val_memmap[:] = M.val[:]
    del M_val_memmap

    M_val_memmap = np.memmap(memmap_path, dtype='float32', mode='r', shape=M.val.shape)
    freq = np.sum(M_val_memmap, axis=0) / (2 * M.iid_count)
    scale = (2 * freq * (1 - freq)).sum()

    Z = M_val_memmap - 2 * freq
    Z_memmap = np.memmap(Z_memmap_path, dtype='float32', mode='w+', shape=Z.shape)
    Z_memmap[:] = Z[:]
    del Z_memmap

    Z_memmap = np.memmap(Z_memmap_path, dtype='float32', mode='r', shape=Z.shape)

    def block_matrix_multiply_optimized_gpu(Z, block_size):
        G = np.memmap(G_memmap_path, dtype='float32', mode='w+', shape=(Z.shape[0], Z.shape[0]))
        for i in tqdm(range(0, Z.shape[0], block_size), desc="Multiplying blocks (G matrix)", unit="block"):
            for j in range(0, Z.shape[0], block_size):
                for k in range(0, Z.shape[1], block_size):
                    G[i:i + block_size, j:j + block_size] += np.dot(Z[i:i + block_size, k:k + block_size],
                                                                    Z[j:j + block_size, k:k + block_size].T)
        return G

    G = block_matrix_multiply_optimized_gpu(Z_memmap, block_size=10000)

    G_memmap = np.memmap(G_memmap_path, dtype='float32', mode='r+', shape=G.shape)
    
    block_size = 10000
    for i in tqdm(range(0, G_memmap.shape[0], block_size), desc="Normalizing G matrix", unit="block"):
        for j in range(0, G_memmap.shape[1], block_size):
            block = G_memmap[i:i + block_size, j:j + block_size]
            block /= scale
            G_memmap[i:i + block_size, j:j + block_size] = block[:]

    del G_memmap
    G_memmap = np.memmap(G_memmap_path, dtype='float32', mode='r', shape=G.shape)
    time2 = time.time()
    time3 = time2 - time1
    print(f"time used: {time3} s")

    return G_memmap

@njit
def inbrec(an, ped, f, avginb):
    s, d = ped[an, 1], ped[an, 2]
    if s <= 0 or d <= 0:
        return avginb[ped[an, 3]]
    else:
        return 0.5 * cffa(s, d, ped, f, avginb)


@njit
def cffa(a1, a2, ped, f, avginb):
    if a1 <= 0 or a2 <= 0:
        return 2 * avginb[max(ped[a1, 3] if a1 > 0 else 0, ped[a2, 3] if a2 > 0 else 0)]
    elif a1 == a2:
        return f[a1] + 1
    else:
        if a1 < a2:
            return 0.5 * (cffa(a1, ped[a2, 1], ped, f, avginb) + cffa(a1, ped[a2, 2], ped, f, avginb))
        else:
            return 0.5 * (cffa(a2, ped[a1, 1], ped, f, avginb) + cffa(a2, ped[a1, 2], ped, f, avginb))


@njit
def calculate_inbreeding(ped, pig_size, max_iterations=10, convergence_threshold=1e-6, block_size=100000):
    n = len(ped)
    f = np.zeros(n, dtype=np.float64)
    avginb = np.zeros(10, dtype=np.float64)

    for iteration in range(max_iterations):
        old_f = f.copy()
        new_avginb = np.zeros(10, dtype=np.float64)
        counts = np.zeros(10, dtype=np.int32)

        for block_start in range(0, n, block_size):
            block_end = min(block_start + block_size, n)
            for i in range(block_start, block_end):
                f[i] = inbrec(i, ped, f, avginb)
                if ped[i, 1] > 0 and ped[i, 2] > 0:
                    new_avginb[ped[i, 3]] += f[i]
                    counts[ped[i, 3]] += 1
            if block_start % 100000 == 0:
                print(f"Diag计算进程：{block_start}/{pig_size}")

        for year in range(10):
            if counts[year] > 0:
                avginb[year] = new_avginb[year] / counts[year]

        if np.max(np.abs(f - old_f)) < convergence_threshold:
            break

    return f


class InbreedingCalculator:
    def __init__(self, csv_file, pig_size):
        data = pd.read_csv(csv_file, header=None, usecols=[0, 1, 2, 4], dtype=int, nrows=pig_size)
        self.ped = data.values
        self.size = len(data)

    def calculate(self):
        return calculate_inbreeding(self.ped, self.size)


def save_inbreeding_coefficients(inbreeding_coefficients, ped, output_file):
    results_df = pd.DataFrame({
        'Animal_ID': ped[:, 0],
        'Inbreeding_Coefficient': inbreeding_coefficients
    })
    results_df.to_csv(output_file, index=False)
    print(f"近交系数已保存到文件: {output_file}")


def build_D_inv2_new(PIG_DATA, pig_size, Diag, max_iterations=10, chunk_size=10000):
    print("开始构建 D_inv2 矩阵...")

    # 处理输入数据
    PIG_DATA = np.hstack((PIG_DATA[:, :3], PIG_DATA[:, 4:5])).astype(int)

    # 创建指示器数组
    mother_ind = PIG_DATA[:, 1] >= 0
    father_ind = PIG_DATA[:, 2] >= 0

    # 初始化对角线元素
    L_diag = np.ones(pig_size)

    # 计算有父母的情况
    m_and_f = mother_ind & father_ind
    if np.any(m_and_f):
        L_diag[m_and_f] = np.sqrt(
            0.5 - 0.25 * (Diag[PIG_DATA[m_and_f, 1].astype(int) - 1] +
                          Diag[PIG_DATA[m_and_f, 2].astype(int) - 1]))

    # 计算只有母亲的情况
    m_but_f = mother_ind & ~father_ind
    if np.any(m_but_f):
        L_diag[m_but_f] = np.sqrt(0.75 - 0.25 * (Diag[PIG_DATA[m_but_f, 1] - 1]))

    # 计算只有父亲的情况
    f_but_m = ~mother_ind & father_ind
    if np.any(f_but_m):
        L_diag[f_but_m] = np.sqrt(0.75 - 0.25 * (Diag[PIG_DATA[f_but_m, 2] - 1]))

    # 确保没有无效值
    L_diag = np.maximum(L_diag, 1e-10)  # 防止除零

    # 构建稀疏对角矩阵
    print("构建稀疏对角矩阵...")
    D_inv2 = sp.diags(1 / L_diag ** 2, format='csr')

    # 检查稀疏度
    sparsity = 1.0 - D_inv2.nnz / (pig_size * pig_size)
    print(f"D_inv2 矩阵稀疏度: {sparsity:.2%}")

    return D_inv2


def build_L_inv(PIG_DATA, pig_size, chunk_size=10000):
    rows, cols, data = [], [], []
    for start in range(0, pig_size, chunk_size):
        end = min(start + chunk_size, pig_size)
        chunk = PIG_DATA[start:end, 1:3]
        for i, (p1, p2) in enumerate(chunk):
            if p1 != -1:
                p1_idx = int(p1) - 1
                if 0 <= p1_idx < pig_size:
                    rows.append(start + i)
                    cols.append(p1_idx)
                    data.append(-0.5)
            if p2 != -1:
                p2_idx = int(p2) - 1
                if 0 <= p2_idx < pig_size:
                    rows.append(start + i)
                    cols.append(p2_idx)
                    data.append(-0.5)

    L_inv = sp.eye(pig_size, format='csr')
    L_inv += sp.csr_matrix((data, (rows, cols)), shape=(pig_size, pig_size))
    return L_inv


def optimize_matrix_operations(D_inv2, L_inv, Dinv2_ag, L_inv_ag, G):
    print("Performing matrix operations...")
    start_time = time.time()

    print("Step 1/4: Computing L_inv.T...")
    L_inv_T = L_inv.T.tocsr()
    print_progress(1, 4, start_time)

    print("Step 2/4: Computing D_inv2.dot(L_inv)...")
    temp = D_inv2.dot(L_inv)
    print_progress(2, 4, start_time)

    print("Step 3/4: Computing A_inv...")
    A_inv = L_inv_T.dot(temp)
    del temp
    gc.collect()
    print_progress(3, 4, start_time)

    A_22 = L_inv_ag.T.dot(Dinv2_ag.dot(L_inv_ag))
    rows, cols = A_inv.shape
    index_G = G.shape[0]
    num = rows - index_G
    G_new = A_inv[num:, num:] - A_22

    print_progress(4, 4, start_time)
    return A_inv[:num, :num], A_inv[:num, num:], G_new


def print_progress(current, total, start_time):
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / current * total
    remaining_time = estimated_total_time - elapsed_time
    print(f"Progress: {current}/{total} steps completed.")
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")
    print(f"Estimated remaining time: {remaining_time:.2f} seconds.")
    print()

def save_state(iteration, xk, residual):
    """
    保存当前迭代状态到文件
    """
    state = {
        'iteration': iteration,
        'xk': xk,
        'residual': residual,
    }
    with open('iteration_state.pkl', 'wb') as f:
        pickle.dump(state, f)
    print(f"已保存迭代状态到 iteration_state.pkl")

def load_state():
    """
    加载上次保存的迭代状态
    """
    try:
        with open('iteration_state.pkl', 'rb') as f:
            state = pickle.load(f)
            print(f"已加载迭代状态，恢复到迭代 {state['iteration']}")
            return state
    except FileNotFoundError:
        print("没有找到保存的迭代状态，重新开始")
        return None


# def load_G_matrix_optimized(G_memmap_path, G_shape, buffer_size=1024 * 1024 * 1024):  # 默认1GB缓冲区
#     """
#     优化的G矩阵加载函数，使用更大的缓冲区和单次读取
#
#     参数:
#         G_memmap_path: G矩阵内存映射文件的路径
#         G_shape: G矩阵的形状 (行数, 列数)
#         buffer_size: 读取缓冲区大小（字节），默认1GB
#     """
#     # 计算需要的总内存大小（字节）
#     total_size = G_shape[0] * G_shape[1] * np.dtype('float32').itemsize
#
#     # 如果总大小小于缓冲区大小，就调整缓冲区大小
#     buffer_size = min(buffer_size, total_size)
#
#     # 创建空数组来存储结果
#     G = np.empty(G_shape, dtype='float32')
#
#     # 使用更大的缓冲区打开文件
#     with open(G_memmap_path, 'rb', buffering=buffer_size) as f:
#         # 创建内存映射，使用较大的缓冲区
#         mm = np.memmap(f, dtype='float32', mode='r', shape=G_shape)
#
#         # 一次性将整个数组复制到内存中
#         # 这比分块读取更快，因为只进行一次I/O操作
#         G[:] = mm[:]
#
#         # 确保数据可写，提高后续运算性能
#         G.flags.writeable = True
#
#     return G

def load_G_matrix_optimized(G_memmap_path, G_shape, buffer_size=1024 * 1024 * 1024):  # 默认1GB缓冲区
    """
    优化的G矩阵加载函数，使用更大的缓冲区和单次读取

    参数:
        G_memmap_path: G矩阵内存映射文件的路径
        G_shape: G矩阵的形状 (行数, 列数)
        buffer_size: 读取缓冲区大小（字节），默认1GB
    """
    # 计算需要的总内存大小（字节）
    # total_size = G_shape[0] * G_shape[1] * np.dtype('float32').itemsize

    # 如果总大小小于缓冲区大小，就调整缓冲区大小
    # buffer_size = min(buffer_size, total_size)

    # 直接使用内存映射文件，不需要显式的open()调用
    mm = np.memmap(G_memmap_path, dtype='float32', mode='r+', shape=G_shape)

    # 创建空数组来存储结果
    # G = np.empty(G_shape, dtype='float32')

    # 将内存映射数据直接复制到数组
    # G[:] = mm[:]

    # 确保数据可写，提高后续运算性能
    # G.flags.writeable = True
    mm.flags.writeable = True

    # return G
    return mm


def save_results(x, filename="1.4GBLUP_4w.csv"):
    print("Saving results...")
    df = pd.DataFrame(x, columns=['Result'])
    df.to_csv(filename, index=False)
    print(f"Result saved to {filename}")
    print("\nPreview of saved data:")
    print(df.head())


def build_and_solve_linear_system(A_11, A_12, G_new, X, Y, Z, G, num_threads=32, k=0.5):
    print("k=", k)
    print("Building linear system...")
    start_time = time.time()
    print(f"线程运行数:{num_threads}")

    # 初始化监控器
    resource_monitor = EnhancedResourceMonitor()
    convergence_monitor = ConvergenceMonitor()

    # 检查是否可以使用GPU
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"Using device: {device}")

    print("Step 1/7: Computing sp_X...")
    resource_monitor.update("Computing sp_X")
    sp_X = sp.csr_matrix(X.T)
    print_progress(1, 7, start_time)

    # 并行化矩阵乘法操作
    def parallel_matrix_mul(A, v, num_threads=None):
        if num_threads is None:
            num_threads = get_optimal_threads(A.shape[0])
        # if use_gpu:
        #     # 使用GPU加速
        #     A_gpu = torch.from_numpy(A.toarray() if sp.issparse(A) else A).to(device)
        #     v_gpu = torch.from_numpy(v).to(device)
        #     result = torch.matmul(A_gpu, v_gpu).cpu().numpy()
        #     return result
        # else:
        # 使用多线程CPU计算
        if sp.issparse(A):
            return A.dot(v)

        # 根据矩阵大小动态调整块大小
        min_chunk_size = 5000
        chunk_size = min(min_chunk_size, len(v) // num_threads)
        chunks = []

        for i in range(0, len(v), chunk_size):
            end = min(i + chunk_size, len(v))
            chunks.append((i, end))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    lambda x: A[x[0]:x[1], :].dot(v),
                    chunk
                )
                for chunk in chunks
            ]
            results = [future.result() for future in futures]

        return np.concatenate(results)

    print("Step 2/7: Computing sp_Z...")
    resource_monitor.update("Computing sp_Z")
    sp_Z = Z
    print_progress(2, 7, start_time)

    print("Step 3/7: Preparing for A_UL...")

    def A_UL_matvec(v):
        # 优化矩阵乘法
        temp = parallel_matrix_mul(sp_X, v)
        return parallel_matrix_mul(sp_X.T, temp)

    A_UL = LinearOperator((sp_X.shape[1], sp_X.shape[1]), matvec=A_UL_matvec)
    resource_monitor.update("Preparing A_UL")
    print_progress(3, 7, start_time)

    print("Step 4/7: Preparing for A_UR...")
    g_size = G.shape[0]
    Z1, Z2 = sp_Z[:, :-g_size], sp_Z[:, -g_size:]
    print(f"sp_X.T shape: {sp_X.T.shape}")
    print(f"Z1 shape: {Z1.shape}, Z2 shape: {Z2.shape}")
    # print(f"XtZ1 shape: {XtZ1.shape}, XtZ2 shape: {XtZ2.shape}")

    # 预计算一些常用的矩阵乘积
    XtZ1 = sp_X.T @ Z1
    XtZ2 = sp_X.T @ Z2

    def A_UR_matvec(v):
        v1, v2 = v[:-g_size], v[-g_size:]
        # 并行计算两部分
        result1 = parallel_matrix_mul(XtZ1, v1)
        temp = parallel_matrix_mul(G, v2)
        result2 = parallel_matrix_mul(XtZ2, temp)
        return result1 + result2

    A_UR = LinearOperator((sp_X.shape[1], sp_Z.shape[1]), matvec=A_UR_matvec)
    resource_monitor.update("Preparing A_UR")
    print_progress(4, 7, start_time)

    print("Step 5/7: Preparing for A_BL and A_BR...")

    def A_BL_matvec(v):
        result1 = parallel_matrix_mul(XtZ1.T, v)
        result2 = parallel_matrix_mul(G, parallel_matrix_mul(XtZ2.T, v))
        return np.hstack((result1, result2))

    A_BL = LinearOperator((sp_Z.shape[1], sp_X.shape[1]), matvec=A_BL_matvec)

    # 预计算一些常用的矩阵乘积
    m_11 = Z1.T @ Z1 + k * A_11
    m_12 = Z1.T @ Z2 + k * A_12

    def A_BR_matvec(v):
        v1, v2 = v[:-g_size], v[-g_size:]
        temp = parallel_matrix_mul(G, v2)

        # 并行计算多个部分
        result1 = parallel_matrix_mul(m_11, v1) + parallel_matrix_mul(m_12, temp) + k * v1
        result2_p1 = parallel_matrix_mul(G, parallel_matrix_mul(m_12.T, v1))
        result2_p2 = parallel_matrix_mul(G, parallel_matrix_mul(Z2.T @ Z2 + k * G_new, temp))
        result2 = result2_p1 + result2_p2 + k * temp

        return np.hstack((result1, result2))

    A_BR = LinearOperator((sp_Z.shape[1], sp_Z.shape[1]), matvec=A_BR_matvec)
    resource_monitor.update("Preparing A_BL and A_BR")
    print_progress(5, 7, start_time)

    print("Step 6/7: Constructing sp_Linearsys...")

    def sp_Linearsys_matvec(v):
        n = sp_X.shape[1]
        v1, v2 = v[:n], v[n:]
        # 并行计算两个主要部分
        # 对大型矩阵进行多线程分块计算
        # num_threads = get_optimal_threads(n)
        num_threads = 16
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 分块计算 A_UL
            future_ul = executor.submit(lambda: A_UL.matvec(v1))
            # 分块计算 A_UR
            future_ur = executor.submit(lambda: A_UR.matvec(v2))
            # 分块计算 A_BL
            future_bl = executor.submit(lambda: A_BL.matvec(v1))
            # 分块计算 A_BR
            future_br = executor.submit(lambda: A_BR.matvec(v2))

            try:
                result1 = future_ul.result() + future_ur.result()
                result2 = future_bl.result() + future_br.result()
            except Exception as e:
                print(f"Error during computation: {e}")
                raise
        return np.hstack((result1, result2 + 100 * v2))

    sp_Linearsys = LinearOperator(
        (sp_X.shape[1] + sp_Z.shape[1], sp_X.shape[1] + sp_Z.shape[1]),
        matvec=sp_Linearsys_matvec
    )
    resource_monitor.update("Constructing Linear System")
    print_progress(6, 7, start_time)

    print("Step 7/7: Computing Linear_b...")
    b_U = parallel_matrix_mul(sp_X.T, Y)
    b_B = parallel_matrix_mul(sp_Z.T, Y)
    b_B[-g_size:] = parallel_matrix_mul(G, b_B[-g_size:])
    b_U = b_U.flatten()
    b_B = b_B.flatten()
    Linear_b = np.concatenate([b_U, b_B])
    resource_monitor.update("Computing Linear_b")
    print_progress(7, 7, start_time)

    print("\n开始求解线性系统...")
    solve_start_time = time.time()
    try:
        def improved_callback(xk):
            """
                改进的回调函数，每隔一定迭代保存状态，并检查是否达到提前终止条件
            """
            iteration = getattr(improved_callback, 'iteration', 0) + 1
            improved_callback.iteration = iteration

            if iteration % 10 == 0:
                current_time = time.time() - solve_start_time
                print(f"\n迭代 {iteration}: 当前已经使用的时间 {current_time:.2f} 秒")

                current_residual = np.linalg.norm(sp_Linearsys.matvec(xk) - Linear_b)
                relative_residual = current_residual / np.linalg.norm(Linear_b)
                convergence_monitor.update(relative_residual)

                print(f"\n迭代 {iteration}:")
                print(f"- 相对残差: {relative_residual:.2e}")

                # 保存状态，每10次迭代保存一次
                save_state(iteration, xk, current_residual)

                # 提前终止条件
                if relative_residual < 1e-6:  # 达到较好精度
                    print("达到目标精度，提前终止迭代")
                    return True

                if iteration > 100:
                    recent_residuals = convergence_monitor.residuals[-100:]
                    improvement = (recent_residuals[0] - recent_residuals[-1]) / recent_residuals[0]
                    print(f"- 最近100次迭代改善率: {improvement:.2%}")

                    # 如果收敛性表现不好，提前终止
                    if improvement < -2.0:  # 恶化超过200%
                        print("检测到收敛性严重恶化，提前终止迭代")
                        return True

                    if iteration > 500 and abs(improvement) < 0.01:  # 改善不明显
                        print("检测到收敛停滞，提前终止迭代")
                        return True

            return False

        # print("构建预处理器...")
        try:
            Linear_b = np.asarray(Linear_b, dtype=np.float64)

            # 使用改进的预处理器
            # M = ImprovedBlockPreconditioner(A_UL, A_BR)
            # print(f"预处理器构建完成。形状: {M.shape}, 数据类型: {M.dtype}")

            # 尝试加载之前的状态
            state = load_state()
            if state:
                iteration = state['iteration']
                xk = state['xk']
                # 如果加载了状态，继续迭代
            else:
                iteration = 0
                xk = np.zeros_like(Linear_b)
            max_iterations = 8000
            print("开始迭代求解...")
            y, info = spla.cg(
                sp_Linearsys,
                Linear_b,
                atol=1e-10,
                maxiter=max_iterations - iteration,  # 剩余迭代次数
                x0=xk, # 设置初始解为之前的解
                callback=improved_callback,  # 使用新的callback函数
            )

            if info < 0:
                raise ValueError(f"CG求解器错误，info={info}")
            elif info > 0:
                print(f"Warning: CG求解器达到最大迭代次数，info={info}")

        except Exception as e:
            print(f"预处理器或求解过程出错: {str(e)}")
            print("尝试不使用预处理器求解...")

            y, info = spla.cg(
                sp_Linearsys,
                Linear_b,
                atol=1e-10,
                maxiter=8000,
                callback=improved_callback
            )

        total_iterations = getattr(improved_callback, 'iteration', 0)
        total_time = time.time() - solve_start_time

        # 保存监控结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # resource_monitor.plot(f'resource_usage_{timestamp}.png')
        # convergence_monitor.plot(f'convergence_{timestamp}.png')

        print(f"\n求解完成:")
        print(f"- 总迭代次数: {total_iterations}")
        print(f"- 总用时: {total_time:.2f}秒")
        print(f"- 平均每次迭代用时: {total_time / total_iterations:.4f}秒")

        if info > 0:
            print(f"Warning: CG求解器达到最大迭代次数但未收敛。Info: {info}")
        elif info < 0:
            print(f"Error: CG求解器遇到错误。Info: {info}")

    except Exception as e:
        print(f"求解线性系统时发生错误: {str(e)}")
        return None, None, -1
    finally:
        # 如果使用了GPU，清理GPU内存
        if use_gpu:
            torch.cuda.empty_cache()

    n = y.size
    num = G.shape[1]
    I = sp.eye(n - num, format='csr')
    ZERO = sp.csr_matrix((n - num, num))

    # 打印矩阵和向量的维度
    print(f"I shape: {I.shape}")
    print(f"ZERO shape: {ZERO.shape}")
    print(f"G shape: {G.shape}")
    print(f"y shape: {y.shape}")

    # 分块读取并计算
    def parallel_matrix_mul_block(I, ZERO, G, y):
        # 将 y 分成两部分
        y1 = y[:I.shape[0]]  # 前 n - num 部分
        y2 = y[I.shape[0]:]  # 后 num 部分

        # 分块矩阵乘法
        result_top = I @ y1 + ZERO @ y2
        result_bottom = ZERO.T @ y1 + G @ y2

        # 合并结果
        result = np.concatenate([result_top, result_bottom])
        return result

    print(
        "*************开始分块矩阵乘法 **************")
    
    x = parallel_matrix_mul_block(I, ZERO, G, y)
    print(
        "*************分块矩阵乘法完成！*************")


    return x, y, info


def main():
    # 创建全局资源监控器
    global_monitor = EnhancedResourceMonitor()

    # 限制 OpenBLAS 线程数
    os.environ["OPENBLAS_NUM_THREADS"] = "32"
    os.environ["OMP_NUM_THREADS"] = "32"
    os.environ["MKL_NUM_THREADS"] = "32"
    os.environ["NUMEXPR_NUM_THREADS"] = "32"

    # 打印线程数

    print(f"当前系统线程数: {psutil.cpu_count()}")



    # 获取CPU核心数
    cpu_count = os.cpu_count()
    print(f"Available CPU cores: {cpu_count}")

    # 设置最大线程数
    max_threads = min(32, cpu_count)
    print(f"Using maximum {max_threads} threads")

    # 配置参数
    # pig_size = 1048575
    # pig_size = 5055000
    #pig_size = 323520
    #pig_size = 10110
    pig_size = 40440

    matrix_size = pig_size

    bed_file = "/home/chuannong1/ZLY/2023/QMSim/r_4w/4w.bed"

    #pig_data_file = "/home/chuannong1/ZLY/2024/BLUP/11.21/p1_data_001.csv"
    #pig_data_file = "p1_data_5055000.csv"    
    pig_data_file = "/home/chuannong1/ZLY/2023/QMSim/r_4w/40440.csv"


    try:
        # 监控初始状态
        global_monitor.update("Initial state")

        # 构建G矩阵
        print("\n开始构建G矩阵...")
        G = building_G_matrix(bed_file=bed_file)
        #G_memmap_path = '/mnt/newdisk/ZLY/G_memmap_file_32w'
        #G_shape = (161760, 161760)
        #G_shape = (323520, 323520)
        #G_shape = (20220, 20220)
        #G = load_G_matrix_optimized(G_memmap_path, G_shape)
        # G = G[4055:, 4055:]
        G_size = G.shape[0]
        global_monitor.update("G matrix construction")
        print(f'The size of G matrix: {G_size}')

        # 读取数据
        print("\n读取数据...")
        PIG_DATA = LargeScaleCSVReader(pig_data_file, (pig_size, 13))
        global_monitor.update("Data loading")

        # 计算近交系数
        print("\n计算近交系数...")
        calculator = InbreedingCalculator(pig_data_file, pig_size)
        Diag = calculator.calculate()
        np.save("diag.npy", Diag)
        global_monitor.update("Inbreeding coefficient calculation")

        # 构建L逆矩阵和D逆矩阵
        print("\n构建L逆矩阵和D逆矩阵...")
        # 构建并监控 L_inv
        L_inv = build_L_inv(PIG_DATA, pig_size)
        monitor_matrix_stats(L_inv, "L_inv")

        # 构建并监控 D_inv2
        D_inv2 = build_D_inv2_new(PIG_DATA, pig_size, Diag)
        monitor_matrix_stats(D_inv2, "D_inv2")
        global_monitor.update("L_inv and D_inv2 construction")

        # 准备AG矩阵
        print("\n准备AG矩阵...")
        temp = PIG_DATA[pig_size - G_size:, :]
        temp[:, :3] -= temp[0, 0] - 1
        L_inv_ag = build_L_inv(temp, G_size)
        D_inv2_ag = build_D_inv2_new(temp, G_size, Diag[-G_size:])
        global_monitor.update("AG matrix preparation")

        # 矩阵优化运算
        print("\n开始矩阵优化运算...")
        A_11, A_12, G_new = optimize_matrix_operations(D_inv2, L_inv, D_inv2_ag, L_inv_ag, G)
        global_monitor.update("Matrix optimization")

        # 准备模型数据
        print("\n准备模型数据...")
        X = PIG_DATA.get_column(3)
        X = np.vstack((X, 1 - X, np.zeros_like(X)))
        Y = PIG_DATA.get_column(9)
        if (-1 in Y):
    	    print("Y列中存在-1值")
        else:
            print("Y列中没有-1值")
        Y[Y == -1] = 0
        Y = Y[Y != -1]
        print(len(Y))
        Z = sp.diags((Y != 0).astype(np.float64), format='csr')
        global_monitor.update("Model data preparation")

        # 求解线性系统
        print("\n开始求解线性系统...")
        x, y, info = build_and_solve_linear_system(A_11, A_12, G_new, X, Y, Z, G,
                                                   num_threads=32,  k=0.5)
        global_monitor.update("Linear system solution")

        if x is not None:
            print("\n保存结果...")
            save_results(x)
            print("\n计算完成！")
            global_monitor.update("Results saving")
        else:
            print("\n计算失败，无结果保存。")

        # 保存全局资源使用情况图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # global_monitor.plot(f'global_resource_usage_{timestamp}.png')

    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        traceback.print_exc()

    finally:
        # 最终资源使用情况
        global_monitor.update("Final state")
        print("\n程序执行结束。")


# if __name__ == "__main__":
#     try:
#         print("\n程序开始执行...")
#         start_time = time.time()
#         main()
#         end_time = time.time()
#         print(f"\n总执行时间: {end_time - start_time:.2f} 秒")
#     except Exception as e:
#         print(f"\n程序执行出错: {str(e)}")
#         traceback.print_exc()
#     finally:
#         print("\n程序执行结束。")

if __name__ == "__main__":
    print("\n程序开始执行...")
    start_time = time.time()
    main()  # 直接调用 main 函数
    end_time = time.time()
    print(f"\n总执行时间: {end_time - start_time:.2f} 秒")

    # 除非在 main 中发生异常，否则这段代码会正常执行
    print("\n程序执行结束。")
