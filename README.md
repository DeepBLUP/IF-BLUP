# IFBLUP - Inversion-Free Mixed Linear Model Solver Implementation

## Project Overview

IFBLUP (Inversion-Free Best Linear Unbiased Prediction) is an efficient mixed linear model solver designed specifically for large-scale genetic evaluation in animal and plant breeding. This tool significantly reduces computational complexity and memory requirements by avoiding direct matrix inversion operations, making it possible to process large-scale datasets.

This implementation supports three main models:
- **ABLUP**: Traditional BLUP model based on pedigree relationship matrix
- **GBLUP**: Genomic selection model based on genomic relationship matrix
- **ssGBLUP**: Single-step genomic selection model, combining pedigree and genomic information

## Features

- **Efficient memory management**: avoid storing the full matrix through sparse matrix and linear operator technology
- **No need for inversion**：innovative algorithm avoids the matrix inversion operation in traditional methods
- **Large-scale data support**：can handle large-scale data sets of hundreds of thousands of bodies
- **Resource monitoring**：built-in resource monitoring module, real-time tracking of CPU and memory usage
- **Diagnostic function**：detailed diagnostic log to help analyze and optimize the calculation process

## Environment requirements

### Dependencies

```
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
numba>=0.54.0
pysnptools>=0.4.36
psutil>=5.8.0
tqdm>=4.62.0
```

### Installation dependencies

```bash
pip install numpy scipy pandas numba pysnptools psutil tqdm
```

## Data requirements

### Pedigree and phenotypic data

Pedigree and phenotypic data should be in CSV format with the following columns (example):
- Individual ID
- Father ID (0 means unknown)
- Mother ID (0 means unknown)
- Gender code
- Generation
- Fixed effect 1
- Fixed effect 2
- Other fixed effects...
- Phenotypic values

### Genotypic data

Genotypic data should use the standard binary PLINK format (.bed, .bim, .fam files). The program will automatically read these files to construct the genomic relationship matrix.

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
