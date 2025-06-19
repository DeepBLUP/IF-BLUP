# IFBLUP - Inversion-Free Mixed Linear Model Solver Implementation

## Project Overview

IFBLUP (Inversion-Free Best Linear Unbiased Prediction) is an efficient mixed linear model solver designed specifically for large-scale genetic evaluation in animal and plant breeding. This tool significantly reduces computational complexity and memory requirements by avoiding direct matrix inversion operations, making it possible to process large-scale datasets.

This implementation supports three main models:
- **ABLUP**: Traditional BLUP model based on pedigree relationship matrix
- **GBLUP**: Genomic selection model based on genomic relationship matrix
- **ssGBLUP**: Single-step genomic selection model, combining pedigree and genomic information

## 特点

- **高效内存管理**：通过稀疏矩阵和线性算子技术，避免存储完整矩阵
- **无需求逆**：创新算法避开了传统方法中的矩阵求逆操作
- **大规模数据支持**：能够处理数十万个体的大规模数据集
- **资源监控**：内置资源监控模块，实时跟踪CPU和内存使用情况
- **诊断功能**：详细的诊断日志，帮助分析和优化计算过程

## 环境要求

### 依赖包

```
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
numba>=0.54.0
pysnptools>=0.4.36
psutil>=5.8.0
tqdm>=4.62.0
```

### 安装依赖

```bash
pip install numpy scipy pandas numba pysnptools psutil tqdm
```

## 数据要求

### 系谱和表型数据

系谱和表型数据应为CSV格式，包含以下列（示例）：
- 个体ID
- 父亲ID（0表示未知）
- 母亲ID（0表示未知）
- 性别代码
- 世代
- 固定效应1
- 固定效应2
- 其他固定效应...
- 表型值

### 基因型数据

基因型数据应使用标准的二进制PLINK格式（.bed, .bim, .fam文件）。程序会自动读取这些文件构建基因组关系矩阵。

## Usage

### 1. Configure Parameters

Modify the following key parameters in the code:

```python
# Select model type: 'ABLUP', 'GBLUP', or 'ssGBLUP'
model_to_run = 'GBLUP'

# Set heritability
h2 = 0.5

# Set file paths
absolute_base_dir = "your_data_directory_path"
bed_file_base_name = "genotype_file_base_name" # without path and extension
pig_data_csv_file_name = "pedigree_phenotype_file.csv" # without path
```

### 2. Run the Program

```bash
python 16w无内存映射.py
```

### 3. Output Files

The program will generate the following output files:
- Estimated Breeding Values (EBV)
- Fixed effect estimates
- Diagnostic log file
- Intermediate calculation results (optional saving)

## Advanced Configuration

### Memory and Performance Optimization

For large-scale datasets, you can adjust the following parameters:

```python
# Set iterative solver parameters
max_iterations = 5000  # Maximum number of iterations
convergence_tolerance = 1e-8  # Convergence tolerance

# Set preconditioner parameters
omega = 0.7  # Preconditioner relaxation factor
```

### Parallel Computing

The program automatically detects system cores and optimizes thread usage, but you can also set it manually:

```python
# Set environment variables to control parallel computing
os.environ["OMP_NUM_THREADS"] = "8"  # Set OpenMP thread count
os.environ["MKL_NUM_THREADS"] = "8"  # Set MKL thread count
```

## Important Notes

- For very large datasets, ensure sufficient disk space for intermediate files
- The program automatically manages memory usage, but for particularly large datasets, it's recommended to run on high-memory machines
- The G matrix will be constructed and saved during the first run, and can be reused in subsequent runs to save time

## Citation

If you use this tool in your research, please cite:

```
Citation information to be added
```

## License

This project is licensed under the [MIT License](LICENSE)