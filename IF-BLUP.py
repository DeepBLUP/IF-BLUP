# -*- coding: utf-8 -*-
"""
IFBLUP - Inversion-Free Implementation of Mixed Linear Model Solver

Author: IFBLUP Development Team
Version: 1.0.0
License: MIT

Project Description:
    IFBLUP (Inversion-Free Best Linear Unbiased Prediction) is an efficient mixed linear model solver,
    designed specifically for large-scale genetic evaluation in animal and plant breeding. This tool
    significantly reduces computational complexity and memory requirements by avoiding direct matrix
    inversion operations, making it possible to process large-scale datasets.

Supported Models:
    - ABLUP: Traditional BLUP model based on pedigree relationship matrix
    - GBLUP: Genomic selection model based on genomic relationship matrix
    - ssGBLUP: Single-step genomic selection model, combining pedigree and genomic information

Key Features:
    - Efficient Memory Management: Avoids storing complete matrices through sparse matrix and linear operator techniques
    - No Inversion Required: Innovative algorithm avoids matrix inversion operations used in traditional methods
    - Large-scale Data Support: Capable of processing datasets with hundreds of thousands of individuals
    - Resource Monitoring: Built-in resource monitoring module to track CPU and memory usage in real-time

Usage:
    1. Configure input file paths and model parameters
    2. Run the program: python 16w无内存映射.py
    3. View output results and diagnostic logs

For detailed documentation, please refer to README.md
"""

# -----------------------------------------------------------------------------
# 1. Basic Library Imports
# -----------------------------------------------------------------------------
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator
import mmap
import os
import sys # For diagnostic logging
import time
import gc
import pandas as pd
from numba import njit # jit can also be used, but njit is usually recommended
# import torch # Reserved for potential future use
from pysnptools.snpreader import Bed # Requires pysnptools installation
import psutil
from datetime import datetime
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, identity # Import sparse matrix formats and identity matrix
# from concurrent.futures import ThreadPoolExecutor # Not used in current version, commented out
import pickle
import traceback
from tqdm import tqdm # Requires tqdm installation

# -----------------------------------------------------------------------------
# 2. Monitoring Module
# -----------------------------------------------------------------------------
class EnhancedResourceMonitor:
    """Enhanced Resource Monitor"""

    def __init__(self):
        """Initialize the monitor"""
        self.cpu_percent = []
        self.memory_used = []  # Actual physical memory used (GB)
        self.memory_percent = []
        self.timestamps = []
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid()) # Get current process
        # Get system information
        try:
            self.cpu_count_logical = psutil.cpu_count(logical=True) # Number of logical cores
            self.total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB
        except Exception as e:
            self.cpu_count_logical = 1
            print(f"Failed to initialize resource monitor: {e}")
            self.total_memory = 0

    def update(self, step_name=""):
        """Record and print current resource usage"""
        print(f"\n--- Resource Usage Report: {step_name} ---")
        current_time = time.time() - self.start_time
        print(f"Time elapsed: {current_time:.2f} seconds")

        # CPU monitoring
        try:
            # Get usage percentage list for each logical CPU
            cpu_percent_list = psutil.cpu_percent(interval=0.1, percpu=True)
            if cpu_percent_list is not None and len(cpu_percent_list) > 0:
                avg_cpu = sum(cpu_percent_list) / len(cpu_percent_list)
                self.cpu_percent.append(avg_cpu)
                print(f"Average CPU usage: {avg_cpu:.2f}%")
            else:
                # If unable to get details for each CPU, try to get overall CPU usage
                overall_cpu = psutil.cpu_percent(interval=0.1)
                if overall_cpu is not None:
                    self.cpu_percent.append(overall_cpu)
                    print(f"Overall CPU usage: {overall_cpu:.2f}%")
                else:
                    print("Unable to get CPU usage details.")
        except Exception as e:
            print(f"Failed to get CPU usage: {e}")


        # Memory monitoring
        try:
            memory_info = self.process.memory_info()
            # Resident Set Size (RSS) is the portion of physical memory actually used by the process
            mem_rss = memory_info.rss / (1024 ** 3) # In GB units
            self.memory_used.append(mem_rss)
            print(f"Current process memory usage (RSS): {mem_rss:.2f} GB")

            # Get overall system memory usage
            mem_sys = psutil.virtual_memory()
            system_mem_info = [
                f"- Total physical memory: {mem_sys.total / (1024 ** 3):.2f} GB",
                f"- Available memory: {mem_sys.available / (1024 ** 3):.2f} GB",
                f"- System usage: {mem_sys.percent}%"
            ]
            # On Linux and similar systems, cache and buffers also occupy memory, understanding them helps to assess memory pressure
            if hasattr(mem_sys, 'cached'): # Linux and similar systems usually have cached memory
                system_mem_info.append(f"- Cache (Cached): {mem_sys.cached / (1024 ** 3):.2f} GB")
            if hasattr(mem_sys, 'buffers'): # Linux and similar systems usually have buffers memory
                system_mem_info.append(f"- Buffers: {mem_sys.buffers / (1024 ** 3):.2f} GB")

            print("System memory details:")
            print("\n".join(system_mem_info))

        except Exception as e:
            print(f"Failed to get memory information: {e}")

        # Detailed CPU time (user mode, kernel mode, IO wait, etc.)
        try:
            # Get percentage of time spent in various CPU states
            cpu_times_val = psutil.cpu_times_percent(interval=0.1) # interval=0.1 means sampling for 0.1 seconds
            if cpu_times_val: # Check if successfully obtained
                print(f"Detailed CPU time usage (%):")
                print(f"- User space: {cpu_times_val.user:.2f}")
                print(f"- System space: {cpu_times_val.system:.2f}")
                print(f"- Idle: {cpu_times_val.idle:.2f}")
                # IO wait time is an important indicator for measuring disk or network bottlenecks
                if hasattr(cpu_times_val, 'iowait'): # iowait is only available on some systems (like Linux)
                    print(f"- IO Wait: {cpu_times_val.iowait:.2f}")
            else:
                print("Unable to get detailed CPU time information.")
        except Exception as e:
            print(f"Failed to get detailed CPU time information: {e}")

        self.timestamps.append(current_time)


def monitor_matrix_stats(matrix, name="Matrix"):
    """Monitor and print basic statistics of a matrix"""
    if matrix is None:
        print(f"\n--- {name} is None ---")
        return

    print(f"\n--- {name} Statistics ---")
    try:
        if sp.issparse(matrix): # Check if it's a Scipy sparse matrix
            nnz = matrix.nnz # Number of non-zero elements
            rows, cols = matrix.shape
            total_elements = rows * cols
            # Sparsity = 1 - (non-zero elements / total elements)
            sparsity = 1.0 - nnz / total_elements if total_elements > 0 else 1.0

            # Estimate memory usage of sparse matrix (data, indices, indptr)
            memory_bytes = 0
            if hasattr(matrix, 'data') and matrix.data is not None:
                memory_bytes += matrix.data.nbytes
            if hasattr(matrix, 'indices') and matrix.indices is not None:
                memory_bytes += matrix.indices.nbytes
            # CSR/CSC has indptr, COO has row/col
            if hasattr(matrix, 'indptr') and matrix.indptr is not None:
                memory_bytes += matrix.indptr.nbytes
            elif hasattr(matrix, 'row') and matrix.row is not None and hasattr(matrix, 'col') and matrix.col is not None:
                 memory_bytes += matrix.row.nbytes + matrix.col.nbytes

            memory_mb = memory_bytes / (1024 * 1024)
            print(f"Format: Sparse ({type(matrix).__name__})")
            print(f"Shape: {matrix.shape}")
            print(f"Non-zero elements (nnz): {nnz:,}") # Using thousands separator
            print(f"Sparsity: {sparsity:.4%}") # Display with higher precision
            print(f"Estimated memory usage (sparse): {memory_mb:.2f} MB")
            print(f"Data type (dtype): {matrix.dtype}")

        elif isinstance(matrix, np.ndarray): # Check if it's a NumPy dense array
            memory_mb = matrix.nbytes / (1024 * 1024)
            print(f"Format: NumPy dense array")
            print(f"Shape: {matrix.shape}")
            print(f"Data type (dtype): {matrix.dtype}")
            print(f"Estimated memory usage (dense): {memory_mb:.2f} MB")

        elif isinstance(matrix, np.memmap): # Check if it's a NumPy memory-mapped file
            memory_mb = matrix.nbytes / (1024 * 1024) # This is the theoretical size of the file on disk
            print(f"Format: NumPy memory-mapped (memmap)")
            print(f"Shape: {matrix.shape}")
            print(f"Data type (dtype): {matrix.dtype}")
            print(f"File size (Memmap): {memory_mb:.2f} MB")
            print(f"File path: {matrix.filename}")

        elif isinstance(matrix, LinearOperator): # Check if it's a SciPy linear operator
            print(f"Format: SciPy linear operator (LinearOperator)")
            print(f"Shape: {matrix.shape}")
            print(f"Data type (dtype): {matrix.dtype}")
            # Linear operators don't store the complete matrix, memory usage is small, mainly depends on the data it relies on
            print(f"Note: LinearOperator doesn't directly store the matrix, memory usage reflects its internal state.")

        else: # Other unknown types
            print(f"Format: Unknown type ({type(matrix)})")
            if hasattr(matrix, 'shape'): print(f"Attempting to get shape: {matrix.shape}")
            if hasattr(matrix, 'dtype'): print(f"Attempting to get data type: {matrix.dtype}")
            if hasattr(matrix, 'nbytes'): print(f"Attempting to get number of bytes: {matrix.nbytes}")

    except Exception as e:
        print(f"Failed to get statistics for matrix '{name}': {e}")
        traceback.print_exc() # Print detailed error stack


# -----------------------------------------------------------------------------
# 3. Determine Optimal CPU Thread Count (Informational, actually set via os.environ)
# -----------------------------------------------------------------------------
def get_optimal_threads(matrix_size, max_threads=None):
    """Dynamically determine the optimal thread count based on matrix size (empirical), or use system core count"""
    try:
        available_logical_cpus = psutil.cpu_count(logical=True)
        available_physical_cpus = psutil.cpu_count(logical=False) # Get physical core count
        # Prioritize physical core count as baseline, as it better reflects parallel computing capability
        base_cpus = available_physical_cpus if available_physical_cpus else available_logical_cpus
        base_cpus = base_cpus if base_cpus else 1 # Ensure at least 1

        if max_threads is None:
            max_threads_limit = base_cpus
        else:
            max_threads_limit = min(max_threads, base_cpus) # Not exceeding physical core count

        # Simple empirical rule based on matrix size (can be adjusted based on actual testing)
        if matrix_size < 1000: optimal_threads = min(4, max_threads_limit)
        elif matrix_size < 5000: optimal_threads = min(8, max_threads_limit)
        elif matrix_size < 20000: optimal_threads = min(16, max_threads_limit)
        else: optimal_threads = max_threads_limit # For large matrices, use all available physical cores

        # Finally ensure not exceeding the maximum thread limit
        final_threads = min(optimal_threads, max_threads_limit)
        # print(f"  Based on matrix size {matrix_size} and available cores {base_cpus} (physical priority), recommended threads: {final_threads}")
        return final_threads
    except Exception as e:
        print(f"Failed to get optimal thread count: {e}. Returning default value 1.")
        return 1

# -----------------------------------------------------------------------------
# 4. Preconditioning Module (Enhanced - Using diag(Z'Z + k*diag(A^-1)) to approximate diagonal)
# -----------------------------------------------------------------------------
class ImprovedBlockPreconditioner(LinearOperator):
    """
    Improved block diagonal preconditioner.
    Uses exact or sampled diagonal of A_UL (S11).
    Uses diag(Z'Z + k*diag(A^-1)) to approximate the diagonal of A_BR (S22).
    """
    def __init__(self, A_UL, A_BR, # S11 and S22 operators
                 Z_for_S22, k_val, diag_A_inv_for_S22, # <-- New parameters
                 omega=0.7, debug=True,
                 sampling_ratio_ul=0.1, min_samples_ul=100, max_samples_ul=5000): # S11 parameters
        """
        Initialize the preconditioner.

        Parameters:
          ... (A_UL, A_BR, omega, debug, sampling_ratio_ul, min_samples_ul, max_samples_ul same as before) ...
          Z_for_S22 (scipy.sparse.csr_matrix): Z matrix corresponding to S22 block (usually the complete Z_solve).
          k_val (float): Variance ratio lambda.
          diag_A_inv_for_S22 (np.ndarray): Pre-calculated diagonal elements vector of A^-1.
        """
        self.dtype = np.float64
        self.n1 = A_UL.shape[0] if A_UL is not None else 0
        self.n2 = A_BR.shape[0] if A_BR is not None else 0 # n2 usually equals n_random
        self.shape = (self.n1 + self.n2, self.n1 + self.n2)
        self.debug = debug
        self.omega = omega

        print(f"--- Improved Block Preconditioner Initialization (Omega={self.omega}) ---")
        print(f"Processing block sizes - S11 (A_UL): {A_UL.shape if A_UL else 'None'}, S22 (A_BR): {A_BR.shape if A_BR else 'None'}")
        if self.n2 > 0:
            print(f"Will use diag(Z'Z + k*diag(A^-1)) to approximate S22 diagonal.")

        self.time_stats = {"setup": 0, "matvec_calls": 0, "total_matvec_time": 0}
        setup_start = time.time()

        self.inv_diag_UL = np.ones(self.n1, dtype=self.dtype)
        self.inv_diag_BR = np.ones(self.n2, dtype=self.dtype)
        eps = max(np.finfo(self.dtype).eps, 1e-10)

        # --- Get diagonal of S11 (A_UL) (logic unchanged) ---
        # (omitting this part of code, same as previous version)
        if A_UL is not None and self.n1 > 0:
            try:
                print("Estimating S11 (A_UL) diagonal...")
                num_samples_ul = max(min_samples_ul, int(self.n1 * sampling_ratio_ul))
                num_samples_ul = min(num_samples_ul, max_samples_ul, self.n1)
                if self.n1 <= max_samples_ul * 1.2:
                     diag_UL_vals = np.zeros(self.n1, dtype=self.dtype)
                     for i in range(self.n1):
                         unit_vec = np.zeros(self.n1, dtype=self.dtype); unit_vec[i] = 1.0
                         diag_UL_vals[i] = A_UL.matvec(unit_vec)[i]
                else:
                     # ... (sampling logic) ...
                     pass # Keep previous sampling logic
                non_pos_mask = diag_UL_vals <= eps
                if np.any(non_pos_mask):
                    print(f"  Warning: Found {np.sum(non_pos_mask)} non-positive values in S11 diagonal, replaced.")
                    diag_UL_vals[non_pos_mask] = eps
                self.inv_diag_UL = 1.0 / diag_UL_vals
            except Exception as e:
                print(f"  Failed to get S11 diagonal: {e}. Using identity diagonal.")
                self.inv_diag_UL = np.ones(self.n1, dtype=self.dtype)
        else:
             print("  S11 (A_UL) is empty or not provided, using identity diagonal.")


        # --- Using diag(Z'Z + k*diag(A^-1)) to approximate the diagonal of S22 (A_BR) ---
        if A_BR is not None and self.n2 > 0:
            try:
                print(f"Using diag(Z'Z + k*diag(A^-1)) to approximate S22 diagonal...")
                if Z_for_S22 is None or Z_for_S22.shape[1] != self.n2:
                    raise ValueError(f"Z matrix column count does not match S22 dimension.")
                if diag_A_inv_for_S22 is None or len(diag_A_inv_for_S22) != self.n2:
                     raise ValueError(f"diag(A^-1) vector length does not match S22 dimension.")

                # 1. Calculate diag(Z'Z)
                print("  Calculating diag(Z'Z)...")
                # Z_for_S22 is (n_obs, n_random)
                diag_ZZ = Z_for_S22.power(2).sum(axis=0).A1 # .A1 converts to 1D array
                if len(diag_ZZ) != self.n2: raise ValueError("Calculated diag(Z'Z) length does not match.")
                print(f"  diag(Z'Z) statistics: Mean={np.mean(diag_ZZ):.4f}, Min={np.min(diag_ZZ):.4f}, Max={np.max(diag_ZZ):.4f}")

                # 2. Calculate approximate diagonal diag_BR_vals = diag(Z'Z) + k * diag(A^-1)
                print(f"  Calculating diag(Z'Z) + k={k_val:.4f} * diag(A^-1)...")
                diag_BR_vals = diag_ZZ + k_val * diag_A_inv_for_S22
                print(f"  Approximate diag(S22) statistics: Mean={np.mean(diag_BR_vals):.4f}, Min={np.min(diag_BR_vals):.4f}, Max={np.max(diag_BR_vals):.4f}")

                # 3. Calculate diagonal inverse, handle non-positive values
                non_pos_mask = diag_BR_vals <= eps
                num_non_pos = np.sum(non_pos_mask)
                if num_non_pos > 0:
                    print(f"  Warning: Found {num_non_pos} non-positive or near-zero values in approximate S22 diagonal, replaced with {eps:.1e}")
                    diag_BR_vals[non_pos_mask] = eps
                self.inv_diag_BR = 1.0 / diag_BR_vals

            except Exception as e:
                print(f"  Error approximating S22 diagonal: {e}. Using identity diagonal.")
                traceback.print_exc()
                self.inv_diag_BR = np.ones(self.n2, dtype=self.dtype)
        else:
            print("  S22 (A_BR) is empty or not provided, using identity diagonal.")

        self.time_stats["setup"] = time.time() - setup_start
        print(f"Preconditioner diagonal preparation complete (using diag(Z'Z + k*diag(A^-1)) to approximate S22). Time: {self.time_stats['setup']:.2f} seconds")
        if self.n1 > 0:
            print(f"  S11 inverse diagonal: Min={np.min(self.inv_diag_UL):.2e}, Max={np.max(self.inv_diag_UL):.2e}, Mean={np.mean(self.inv_diag_UL):.2e}")
        if self.n2 > 0:
            # Mean and range here are based on (diag(Z'Z + k*diag(A^-1)))^-1
            print(f"  S22 (approximate) inverse diagonal: Min={np.min(self.inv_diag_BR):.2e}, Max={np.max(self.inv_diag_BR):.2e}, Mean={np.mean(self.inv_diag_BR):.2e}")

    # _matvec and _rmatvec methods remain unchanged
    def _matvec(self, x):
        # ... (code same as before) ...
        matvec_start = time.time()
        self.time_stats["matvec_calls"] += 1
        x_arr = np.asarray(x).ravel()
        if x_arr.shape[0] != self.shape[1]: raise ValueError("Input vector shape mismatch")
        result = np.zeros_like(x_arr, dtype=self.dtype)
        if self.n1 > 0:
            x1 = x_arr[:self.n1]
            scaled_x1 = x1 * self.inv_diag_UL
            result[:self.n1] = self.omega * scaled_x1 + (1 - self.omega) * x1
        if self.n2 > 0:
            x2 = x_arr[self.n1:]
            scaled_x2 = x2 * self.inv_diag_BR
            result[self.n1:] = self.omega * scaled_x2 + (1 - self.omega) * x2
        # ... (debug print logic) ...
        matvec_time = time.time() - matvec_start
        self.time_stats["total_matvec_time"] += matvec_time
        return result

    def _rmatvec(self, x):
        return self._matvec(x)


# -----------------------------------------------------------------------------
# 5. Data Reading and Construction Module
# -----------------------------------------------------------------------------
class LargeScaleCSVReader:
    """
    Memory-mapped reader for large CSV files.
    Allows accessing specific rows/columns via slicing or indexing like a Numpy array,
    without loading the entire file into memory.
    Note: Requires specific CSV format, e.g., comma-separated, no complex quoted characters.
    """
    def __init__(self, filename, shape, dtype=np.float64, delimiter=',', encoding='utf-8', has_header=True):
        """
        Initializes the reader.

        Parameters:
        filename (str): Path to the CSV file.
        shape (tuple): Expected shape of the data (number of rows, number of columns).
                       Number of rows can be an estimate, number of columns needs to be accurate.
        dtype (np.dtype): Data type.
        delimiter (str): Delimiter of the CSV file.
        encoding (str): File encoding.
        has_header (bool): Whether the file has a header row.
        """
        self.filename = filename
        self.shape = shape # (estimated rows, accurate columns)
        self.dtype = dtype
        self.delimiter = delimiter
        self.encoding = encoding
        self.has_header = has_header
        self.num_cols = shape[1] # Store number of columns

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found: {filename}")
        try:
            self.file_size = os.path.getsize(filename)
            if self.file_size == 0:
                print(f"Warning: Data file {filename} is empty.")
        except Exception as e:
            print(f"Could not get file size {filename}: {e}")
            self.file_size = -1

        # Pre-calculate start position of each line (optional, may be time-consuming for large files, but speeds up subsequent reads)
        # self._line_offsets = self._calculate_line_offsets()

    def _calculate_line_offsets(self):
        """(Optional) Calculates byte offsets of each line in the file"""
        print(f"  Calculating line offsets for {self.filename}...")
        offsets = [0] # Offset of the first line is always 0 (or after header)
        try:
            with open(self.filename, 'rb') as f: # Read bytes in binary mode
                if self.has_header:
                    f.readline() # Skip header
                    offsets = [f.tell()] # Offset of the first data line

                while f.readline(): # Read each line to find the start of the next
                    offsets.append(f.tell())
            # The last offset is the end of the file, not needed
            if len(offsets)>1 : offsets.pop()
            print(f"  Calculation complete, found {len(offsets)} data lines.")
            return offsets
        except Exception as e:
            print(f"  Failed to calculate line offsets: {e}. Will read line by line.")
            return None

    def _parse_line(self, line_str, col_key):
        """Parses a single line of data, extracting required data based on column index"""
        try:
            # Split line data
            parts = line_str.strip().split(self.delimiter)
            num_parts = len(parts)

            # Extract data based on column index
            if isinstance(col_key, int): # Single column index
                if col_key < 0: col_key += self.num_cols # Support negative indexing
                if 0 <= col_key < num_parts:
                    return self.dtype(parts[col_key])
                else:
                    return self.dtype(np.nan) # Column index out of bounds, return NaN
            elif isinstance(col_key, slice): # Column slice
                # Parse slice parameters
                start = col_key.start if col_key.start is not None else 0
                stop = col_key.stop if col_key.stop is not None else self.num_cols
                step = col_key.step if col_key.step is not None else 1
                # Apply slice and convert type, handle out-of-bounds indices
                return np.array([self.dtype(parts[c]) if 0 <= c < num_parts else np.nan
                                 for c in range(start, stop, step)], dtype=self.dtype)
            elif isinstance(col_key, (list, np.ndarray)): # List of column indices
                # Extract specified columns, handle out-of-bounds indices
                 # Handle negative indices
                cols = [c + self.num_cols if c < 0 else c for c in col_key]
                return np.array([self.dtype(parts[c]) if 0 <= c < num_parts else np.nan
                                 for c in cols], dtype=self.dtype)
            else: # Unsupported column index type
                raise TypeError(f"Unsupported column index type: {type(col_key)}")

        except ValueError as ve: # Data conversion failed
            # print(f"    Warning: In-line data conversion failed - {ve}. May return NaN.")
            # Return NaN or original parts as appropriate
            if isinstance(col_key, int): return self.dtype(np.nan)
            # For slices or lists, attempt partial conversion, fill failed parts with NaN (more complex, not implemented for now)
            # Simplified handling: if any part fails conversion, the entire result might be problematic or return NaN array
            if isinstance(col_key, slice):
                 s_len = len(range(col_key.start or 0, col_key.stop or self.num_cols, col_key.step or 1))
                 return np.full(s_len, np.nan, dtype=self.dtype)
            elif isinstance(col_key, (list, np.ndarray)):
                 return np.full(len(col_key), np.nan, dtype=self.dtype)
            else:
                 return self.dtype(np.nan) # Fallback
        except Exception as e: # Other parsing errors
            print(f"    Unknown error occurred while parsing line: {e}, line content: '{line_str[:100]}...'")
            # Return NaN to indicate error
            if isinstance(col_key, int): return self.dtype(np.nan)
            if isinstance(col_key, slice):
                 s_len = len(range(col_key.start or 0, col_key.stop or self.num_cols, col_key.step or 1))
                 return np.full(s_len, np.nan, dtype=self.dtype)
            elif isinstance(col_key, (list, np.ndarray)):
                 return np.full(len(col_key), np.nan, dtype=self.dtype)
            else:
                 return self.dtype(np.nan)

    def __getitem__(self, key):
        """Gets data via indexing or slicing"""
        if self.file_size == 0: # Handle empty file case
             # Return empty array or None based on requested shape
             # (complex logic for empty file returning empty array omitted, as files are usually non-empty)
             print("Error: File is empty, cannot read data.")
             return None

        # Parse row and column indices/slices
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
        elif isinstance(key, (int, slice)): # If only one index is provided, assume it's a row index, get all columns
            row_key, col_key = key, slice(None)
        else:
            raise TypeError("Index must be an integer, slice, or a tuple containing two elements (row, col).")

        # --- Handle row indexing/slicing ---
        try:
            with open(self.filename, 'r', encoding=self.encoding) as f:
                # Attempt memory mapping (read-only)
                try:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                except ValueError: # File might be empty or unmappable
                     print(f"Warning: File {self.filename} cannot be memory-mapped, attempting standard file read.")
                     mm = None # Mark memory mapping as unused

                try:
                    # Handle single row index
                    if isinstance(row_key, int):
                        if row_key < 0: row_key += self.shape[0] # Support negative indexing (based on expected rows)
                        if not (0 <= row_key < self.shape[0]): raise IndexError("Row index out of bounds")

                        if mm: # Use memory mapping
                             # Need to find start and end positions of corresponding line (if offsets not pre-calculated, needs scanning)
                             # Simplified: assume lines are roughly uniform, estimate position and search for newlines around it (inefficient)
                             # More reliable method is to read line by line, or pre-calculate offsets
                             mm.seek(0) # Go to start of file
                             if self.has_header: mm.readline() # Skip header
                             for _ in range(row_key): # Read until target line
                                 if not mm.readline(): raise IndexError("File ended prematurely while reading line")
                             line_bytes = mm.readline()
                             if not line_bytes: raise IndexError("Could not read target line")
                             return self._parse_line(line_bytes.decode(self.encoding), col_key)
                        else: # Use standard file read
                             if self.has_header: f.readline() # Skip header
                             for _ in range(row_key):
                                 if not f.readline(): raise IndexError("File ended prematurely while reading line")
                             line_str = f.readline()
                             if not line_str: raise IndexError("Could not read target line")
                             return self._parse_line(line_str, col_key)

                    # Handle row slice
                    elif isinstance(row_key, slice):
                        # Parse row slice parameters
                        start_row = row_key.start if row_key.start is not None else 0
                        stop_row = row_key.stop if row_key.stop is not None else self.shape[0] # Use expected number of rows
                        step_row = row_key.step if row_key.step is not None else 1

                        if start_row < 0: start_row += self.shape[0]
                        if stop_row < 0: stop_row += self.shape[0]
                        # Ensure range is valid
                        start_row = max(0, start_row)
                        stop_row = min(self.shape[0], stop_row)

                        data_rows = []
                        current_row_idx = 0
                        if mm: mm.seek(0)
                        else: f.seek(0)

                        if self.has_header: # Skip header
                            if mm: mm.readline()
                            else: f.readline()

                        while current_row_idx < stop_row: # Iterate until stop row is reached
                            if mm: line_bytes = mm.readline()
                            else: line_str = f.readline()

                            if not (line_bytes if mm else line_str): break # End of file

                            # Check if it's a line to be extracted
                            if current_row_idx >= start_row and (current_row_idx - start_row) % step_row == 0:
                                line_to_parse = line_bytes.decode(self.encoding) if mm else line_str
                                parsed_data = self._parse_line(line_to_parse, col_key)
                                data_rows.append(parsed_data)

                            current_row_idx += 1
                            # If slice stop point is reached, can exit loop early
                            if current_row_idx >= stop_row : break

                        # Stack collected row data into a Numpy array
                        if data_rows:
                             # If result is a list of scalars (single column selection), convert directly
                             if isinstance(col_key, int):
                                 return np.array(data_rows, dtype=self.dtype)
                             # If result is a list of arrays (multiple column selection), try stacking
                             else:
                                 try:
                                     return np.vstack(data_rows) # More robust stacking
                                 except ValueError:
                                      # If row lengths are inconsistent, may need special handling or error
                                      print("Warning: Row slice results have inconsistent lengths, possibly due to parsing errors or irregular data. Returning object array.")
                                      return np.array(data_rows, dtype=object)

                        else: # If no rows were selected
                             # Return empty array of correct shape based on column selection
                             if isinstance(col_key, int): return np.empty((0,), dtype=self.dtype)
                             if isinstance(col_key, slice):
                                 num_cols_selected = len(range(col_key.start or 0, col_key.stop or self.num_cols, col_key.step or 1))
                                 return np.empty((0, num_cols_selected), dtype=self.dtype)
                             if isinstance(col_key, (list, np.ndarray)):
                                 return np.empty((0, len(col_key)), dtype=self.dtype)
                             return np.empty((0, self.num_cols), dtype=self.dtype) # Fallback

                    else:
                        raise TypeError(f"Unsupported row index type: {type(row_key)}")

                finally:
                    if mm: mm.close() # Close memory map

        except FileNotFoundError:
             print(f"Error: File {self.filename} not found.")
             raise
        except Exception as e:
             print(f"Error occurred while reading file {self.filename}: {e}")
             traceback.print_exc()
             raise


    def get_column(self, col_index):
        """Efficiently gets single column data"""
        # Implemented using __getitem__, passing row slice slice(None) and column index
        try:
            return self[slice(None), col_index]
        except Exception as e:
            print(f"Error getting column {col_index}: {e}")
            # Return an empty array or handle error as needed
            return np.array([], dtype=self.dtype)

# -----------------------------------------------------------------------------
# Helper Functions (Progress Bar, State Save/Load)
# -----------------------------------------------------------------------------

def print_progress(current, total, start_time_prog, bar_length=50):
    """Prints progress information with a progress bar and estimated time"""
    if total == 0: # Prevent division by zero
        print("Total tasks are zero, cannot display progress.")
        return

    fraction = current / total
    arrow = '#' * int(round(fraction * bar_length))
    spaces = ' ' * (bar_length - len(arrow))

    elapsed = time.time() - start_time_prog
    # Estimated Time Remaining (ETA)
    if fraction > 1e-6: # Avoid early division by zero or unstable estimates
        est_total_time = elapsed / fraction
        eta = est_total_time - elapsed
        eta_str = f"{eta:.1f}s" if eta >= 0 else "N/A"
    else:
        eta_str = "N/A"

    # Use \r carriage return to update progress bar in place
    print(f"\rProgress: [{arrow + spaces}] {int(fraction * 100)}% ({current}/{total}) | Elapsed: {elapsed:.1f}s | ETA: {eta_str}", end='')
    # Print newline when complete
    if current == total:
        print() # Newline

def save_state(iteration_s, xk_s, residual_s, state_file_s='iteration_state.pkl'):
    """Saves iteration state to a Pickle file"""
    try:
        # Ensure solution vector is a standard Numpy array for serialization
        xk_to_save = np.asarray(xk_s)
        state = {'iteration': iteration_s, 'xk': xk_to_save, 'residual': residual_s}
        # Open file in 'wb' (write binary) mode
        with open(state_file_s, 'wb') as f:
            # Use pickle to save dictionary object, choose highest protocol for efficiency and compatibility
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        # print(f"  Iteration state saved to {state_file_s} (iteration: {iteration_s})") # Can uncomment for more detailed logs
    except Exception as e:
        print(f"\nWarning: Failed to save iteration state to {state_file_s}: {e}")
        traceback.print_exc()

def load_state(state_file_l='iteration_state.pkl'):
    """Loads iteration state from a Pickle file"""
    try:
        if not os.path.exists(state_file_l):
            # print(f"State file {state_file_l} does not exist, will start from scratch.")
            return None # File does not exist, return None
        # Open file in 'rb' (read binary) mode
        with open(state_file_l, 'rb') as f:
            state = pickle.load(f) # Load saved state dictionary
            loaded_iter = state.get('iteration', 'N/A')
            loaded_xk = state.get('xk')
            print(f"Successfully loaded state from {state_file_l} (iteration: {loaded_iter})")
            if loaded_xk is not None:
                print(f"  Loaded solution vector 'xk' shape: {loaded_xk.shape}, type: {type(loaded_xk)}")
            else:
                 print("  Warning: Solution vector 'xk' not found in loaded state.")
            return state
    except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e_load:
        print(f"\nWarning: Error occurred while loading state file {state_file_l}: {e_load}")
        print("  This might be due to file corruption, incompleteness, or incompatible format.")
        # Try to delete potentially corrupted file to avoid errors next time
        if os.path.exists(state_file_l):
            try:
                os.remove(state_file_l)
                print(f"  Attempted to delete corrupted state file: {state_file_l}")
            except OSError as e_remove:
                print(f"  Failed to delete file {state_file_l}: {e_remove}")
        return None # Return None to indicate loading failure
    except Exception as e: # Catch other possible exceptions
         print(f"\nWarning: Unknown error occurred while loading state file {state_file_l}: {e}")
         traceback.print_exc()
         return None

# -----------------------------------------------------------------------------
# 5. Data Reading and Construction Module (Continued)
# -----------------------------------------------------------------------------

# --- G Matrix Construction Related Functions ---

def debug_verify_G_construction(Z_centered, scale_val, G_result, sample_size=100):
    """
    [Debug Function] Verifies the correctness of G matrix construction (G = Z Z' / scale).
    Compares elements in G matrix with values directly calculated from Z matrix by random sampling.

    Parameters:
    Z_centered (np.ndarray or np.memmap): Centered genotype matrix (individuals x SNPs).
    scale_val (float): Scaling factor (e.g., 2 * sum(p*(1-p))).
    G_result (np.ndarray or np.memmap): Constructed G matrix.
    sample_size (int): Number of individuals to sample for verification.
    """
    print("\n[Debug] Verifying G matrix construction process...")
    if Z_centered is None or G_result is None:
        print("  Error: Z centered matrix or G result matrix is None, cannot verify.")
        return

    n_individuals, n_snps = Z_centered.shape
    if n_individuals != G_result.shape[0] or n_individuals != G_result.shape[1]:
        print(f"  Error: Dimensions of Z({Z_centered.shape}) and G({G_result.shape}) do not match.")
        return
    if scale_val == 0:
        print("  Warning: Scaling factor scale_val is 0, verification results might be meaningless.")
        # Or return directly, as G calculation will involve division by zero
        # return

    # Ensure sample_size does not exceed number of individuals
    sample_size = min(sample_size, n_individuals)
    if sample_size <= 1 :
         print("  Sample size is too small for effective sampling verification.")
         return

    # Randomly select sample indices
    sample_indices = np.random.choice(n_individuals, sample_size, replace=False)

    max_abs_error = 0.0
    max_rel_error = 0.0
    max_abs_err_coords = (-1, -1)
    max_rel_err_coords = (-1, -1)
    errors = [] # Store relative errors

    print(f"  Sampling {sample_size} individuals for verification...")
    for i_idx, i in enumerate(tqdm(sample_indices, desc="G Verification Sampling", leave=False, ncols=80)):
        # Only calculate upper triangle (including diagonal) as G is symmetric
        for j_idx, j in enumerate(sample_indices):
            if i > j: continue # G[i,j] == G[j,i]

            try:
                # Get corresponding rows from Z matrix (ensure data loaded into memory)
                Z_i_row = np.array(Z_centered[i, :])
                Z_j_row = np.array(Z_centered[j, :])

                # Directly calculate expected value of G[i, j]
                zij_dot = np.dot(Z_i_row, Z_j_row)
                # Handle case where scale_val might be zero
                expected_gij = zij_dot / scale_val if scale_val != 0 else np.inf

                # Get actual constructed G[i, j] value
                actual_gij = G_result[i, j]

                # Calculate absolute and relative errors
                abs_error = np.abs(actual_gij - expected_gij)
                rel_error = abs_error / abs(expected_gij) if abs(expected_gij) > 1e-10 else (abs_error if abs(actual_gij) > 1e-10 else 0.0)
                errors.append(rel_error)

                # Update max error records
                if abs_error > max_abs_error:
                    max_abs_error = abs_error
                    max_abs_err_coords = (i, j)
                if rel_error > max_rel_error:
                    max_rel_error = rel_error
                    max_rel_err_coords = (i, j)

            except Exception as e:
                print(f"\n  Error calculating or comparing G[{i},{j}]: {e}")
                errors.append(np.inf) # Mark as error

    if errors:
        valid_errors = [e for e in errors if e != np.inf]
        if valid_errors:
             mean_rel_error = np.mean(valid_errors)
             median_rel_error = np.median(valid_errors)
             print(f"  G matrix verification results (based on {len(valid_errors)} valid sample pairs):")
             print(f"    Average relative error: {mean_rel_error:.6e}")
             print(f"    Median relative error: {median_rel_error:.6e}")
             print(f"    Max absolute error: {max_abs_error:.6e} at G[{max_abs_err_coords[0]},{max_abs_err_coords[1]}]")
             print(f"    Max relative error: {max_rel_error:.6e} at G[{max_rel_err_coords[0]},{max_rel_err_coords[1]}]")

             # Give warning based on error magnitude
             if max_rel_error > 1e-5 or max_abs_error > 1e-5:
                 print("  Warning: G matrix construction may have significant numerical precision issues. Please check Z matrix centering and scaling factor calculation.")
                 # Print detailed information at max error location
                 i, j = max_rel_err_coords
                 if i >= 0 and j >= 0:
                      try:
                           Z_i_row = np.array(Z_centered[i, :])
                           Z_j_row = np.array(Z_centered[j, :])
                           zij_dot = np.dot(Z_i_row, Z_j_row)
                           expected_gij = zij_dot / scale_val if scale_val != 0 else np.inf
                           actual_gij = G_result[i, j]
                           print(f"    Max relative error example: G[{i},{j}] = {actual_gij:.6f} vs expected {expected_gij:.6f} (Z_i . Z_j = {zij_dot:.4f})")
                      except Exception as e_detail:
                           print(f"    Error getting max error details: {e_detail}")
             else:
                 print("  G matrix construction verification passed (errors within acceptable range).")
        else:
             print("  Could not perform effective G matrix construction verification (all sample pair calculations failed).")
    else:
        print("  Could not perform G matrix construction verification (no sample pairs).")

import numpy as np
import os
import time
import gc
from tqdm import tqdm
import traceback
from pysnptools.snpreader import Bed # Ensure imported

# Assume monitor_matrix_stats and debug_verify_G_construction functions are defined elsewhere
# from your_module import monitor_matrix_stats, debug_verify_G_construction
# If not, copy or import them from previous code

def building_G_matrix(bed_file,
                      Z_memmap_path='Z_memmap_file.dat', # Z memmap file path (this version no longer creates or relies on it)
                      G_memmap_path='G_memmap_file.dat', # G matrix memmap file path
                      block_size_G=81000,                # Individual block size for G matrix accumulation
                      snp_block_size=51000,              # Number of SNPs to process at a time
                      overwrite=False,                  # Whether to overwrite existing G file
                      impute_method='mean',             # Genotype missing value imputation method ('mean' or 'zero')
                      target_dtype=np.float64):         # Target data type (float64 or float32)
    print(f"\n--- Starting G matrix construction [in-memory mode] (dtype={target_dtype.__name__}, SNP block size={snp_block_size}, Missing imputation: {impute_method}) ---") # Modified print message
    start_time_g_build = time.time()

    # 1. Get BED file basic information (unchanged)
    try:
        bed_reader = Bed(bed_file, count_A1=False)
        num_individuals = bed_reader.iid_count
        num_snps = bed_reader.sid_count
        print(f"Read from Bed file: {num_individuals} individuals, {num_snps} SNPs")
        if num_individuals <= 0 or num_snps <= 0:
            raise ValueError("Invalid number of individuals or SNPs.")
    except Exception as e:
        print(f"Failed to read BED file information: {e}")
        traceback.print_exc()
        return None

    G_shape = (num_individuals, num_individuals)

    # 2. Remove logic for loading existing G_memmap, as we build in memory
    # if os.path.exists(G_memmap_path) and not overwrite:
    #     ... (this part removed or commented out) ...
    # else:
    #     print(f"Will build new G matrix in memory.")

    # --- If G needs to be built --- (now always needed, as not persisted in memmap)

    # 3. Block-wise calculation of allele frequencies p and scale (this logic unchanged)
    print(f"\nBlock-wise calculation of allele frequencies (SNP block size: {snp_block_size})...")
    p_values = np.zeros(num_snps, dtype=np.float64)
    sum_A1_total = np.zeros(num_snps, dtype=np.float64)
    n_alleles_total = np.zeros(num_snps, dtype=np.int64)
    # ... (p_values, scale_val calculation logic remains unchanged) ...
    # Ensure this part of the code executes correctly
    try:
        bed_reader_freq = Bed(bed_file, count_A1=False)
        for k_start in tqdm(range(0, num_snps, snp_block_size), desc="Calculating Frequencies", ncols=80):
            k_end = min(k_start + snp_block_size, num_snps)
            snp_indices_to_read = np.s_[k_start:k_end]
            bed_reader_subset = bed_reader_freq[:, snp_indices_to_read]
            snp_data = bed_reader_subset.read(dtype=np.float64)
            M_block_freq = snp_data.val # Use different variable name to avoid confusion with subsequent M_block
            
            nan_mask_block_freq = np.isnan(M_block_freq)
            n_alleles_total[k_start:k_end] = 2 * np.sum(~nan_mask_block_freq, axis=0)
            sum_A1_total[k_start:k_end] = np.nansum(M_block_freq, axis=0)

            del M_block_freq, snp_data, nan_mask_block_freq, bed_reader_subset
            gc.collect()

        valid_counts_mask = n_alleles_total > 0
        p_values[valid_counts_mask] = sum_A1_total[valid_counts_mask] / n_alleles_total[valid_counts_mask]
        p_values[~valid_counts_mask] = 0.5
        if np.any(~valid_counts_mask):
            print(f"  Warning: {np.sum(~valid_counts_mask)} SNPs are completely missing, frequencies set to 0.5.")

        scale_val = np.sum(2 * p_values * (1 - p_values))
        if abs(scale_val) < 1e-10:
            print("  Warning: G matrix scaling factor scale is close to zero. Using scale = 1.0 instead.")
            scale_val = 1.0
        print(f"  Allele frequencies and scaling factor calculation complete. Scale = {scale_val:.6f}")
        del sum_A1_total, n_alleles_total, bed_reader_freq
        gc.collect()
    except Exception as e_freq:
        print(f"Error calculating frequencies or scaling factor: {e_freq}")
        traceback.print_exc()
        return None
    # --- p_values and scale_val calculation end ---


    # 4. Initialize G matrix as an in-memory np.ndarray
    print(f"\nInitializing G matrix in memory, shape: {G_shape}...")
    try:
        G_matrix_in_memory = np.zeros(G_shape, dtype=target_dtype) # Modification point
    except MemoryError:
        print(f"Error: Insufficient memory to create G matrix of shape {G_shape} (approx {G_shape[0]*G_shape[1]*target_dtype().itemsize / (1024**3):.2f} GB).")
        print("Please consider reducing the number of individuals or switching back to memory mapping.")
        return None
    except Exception as e:
        print(f"Failed to create in-memory G matrix: {e}")
        return None

    # 5. Block-wise accumulation of G (operation target becomes G_matrix_in_memory)
    print(f"\nBlock-wise accumulating G matrix (SNP block size: {snp_block_size}, individual block size: {block_size_G})...")
    try:
        num_snp_blocks = (num_snps + snp_block_size - 1) // snp_block_size
        num_indi_blocks_G = (num_individuals + block_size_G - 1) // block_size_G
        bed_reader_g = Bed(bed_file, count_A1=False)

        with tqdm(total=num_snp_blocks, desc="SNP Blocks for G", ncols=80) as pbar_snp:
            for k_block_idx in range(num_snp_blocks):
                k_start = k_block_idx * snp_block_size
                k_end = min(k_start + snp_block_size, num_snps)
                
                # --- a. Read current SNP block M_block (unchanged) ---
                try:
                    snp_indices_to_read = np.s_[k_start:k_end]
                    bed_reader_subset = bed_reader_g[:, snp_indices_to_read]
                    snp_data = bed_reader_subset.read(dtype=target_dtype)
                    M_block = snp_data.val
                except MemoryError:
                     print(f"\nInsufficient memory to load SNP block [{k_start}:{k_end}] ({num_individuals} x {k_end-k_start}).")
                     del G_matrix_in_memory, bed_reader_g; gc.collect(); return None # Clean up and return
                except Exception as e_read_block:
                     print(f"\nError reading SNP block [{k_start}:{k_end}]: {e_read_block}")
                     del G_matrix_in_memory, bed_reader_g; gc.collect(); return None

                # --- b. Calculate centered block Z_block (unchanged) ---
                p_values_block = p_values[k_start:k_end]
                nan_mask_block = np.isnan(M_block)
                if impute_method == 'mean':
                     M_block_imp = np.copy(M_block)
                     for j_local in range(M_block.shape[1]):
                          nan_mask_j = nan_mask_block[:, j_local]
                          if np.any(nan_mask_j):
                               M_block_imp[nan_mask_j, j_local] = 2 * p_values_block[j_local]
                     Z_block = M_block_imp - 2 * p_values_block[np.newaxis, :]
                     del M_block_imp
                elif impute_method == 'zero':
                     Z_block_raw = M_block - 2 * p_values_block[np.newaxis, :]
                     Z_block = np.nan_to_num(Z_block_raw, nan=0.0)
                     del Z_block_raw
                else: # Should be 'none' or similar if you want to allow NaNs, but this is risky
                     Z_block = M_block - 2 * p_values_block[np.newaxis, :]
                     if np.any(nan_mask_block): print("Warning: Z_block may contain NaN!")
                
                del M_block, snp_data, nan_mask_block # bed_reader_subset will be handled automatically on next loop or exit
                gc.collect()

                # --- c. Block-wise calculate Z_block @ Z_block.T and accumulate into G_matrix_in_memory ---
                for i_block_idx in range(num_indi_blocks_G):
                    i_start = i_block_idx * block_size_G
                    i_end = min(i_start + block_size_G, num_individuals)
                    Z_block_i = None
                    try:
                        Z_block_i = Z_block[i_start:i_end, :]
                    except Exception as e:
                        print(f"\n  Error getting Z_block_i (rows {i_start}:{i_end}): {e}")
                        continue

                    for j_block_idx in range(i_block_idx, num_indi_blocks_G):
                        j_start = j_block_idx * block_size_G
                        j_end = min(j_start + block_size_G, num_individuals)
                        block_contrib = None
                        Z_block_j = None # Ensure Z_block_j is reset at start of each j loop

                        try:
                            if i_block_idx == j_block_idx:
                                block_contrib = Z_block_i @ Z_block_i.T
                                G_matrix_in_memory[i_start:i_end, j_start:j_end] += block_contrib # Modification point
                            else:
                                Z_block_j = Z_block[j_start:j_end, :]
                                block_contrib = Z_block_i @ Z_block_j.T
                                G_matrix_in_memory[i_start:i_end, j_start:j_end] += block_contrib # Modification point
                                G_matrix_in_memory[j_start:j_end, i_start:i_end] += block_contrib.T # Modification point
                        except Exception as e_inner:
                             print(f"\n  Error calculating or accumulating G block ({i_block_idx}, {j_block_idx}): {e_inner}")
                        finally:
                             if block_contrib is not None: del block_contrib
                             if Z_block_j is not None: del Z_block_j # Clean up Z_block_j
                    
                    if Z_block_i is not None: del Z_block_i
                    gc.collect()
                
                del Z_block
                gc.collect()

                # No longer need G_memmap.flush()
                pbar_snp.update(1)
        del bed_reader_g
    except Exception as e_accum:
        print(f"\nError accumulating G matrix: {e_accum}")
        traceback.print_exc()
        del G_matrix_in_memory; gc.collect(); return None # Modification point

    # --- 6. Final scaling of G matrix (operation target becomes G_matrix_in_memory) ---
    print(f"\nPerforming final scaling of G matrix (dividing by {scale_val:.6f})...")
    try:
        if abs(scale_val) > 1e-10:
            # For in-memory ndarray, can operate directly, no need for blocking unless to reduce peak memory (but G is already in memory here)
            G_matrix_in_memory /= scale_val # Modification point (direct operation on entire matrix)
        else:
            print("  Warning: scale_val is close to zero, skipping G matrix scaling step.")
    except MemoryError:
         print(f"\nInsufficient memory for G matrix scaling operation.")
         del G_matrix_in_memory; gc.collect(); return None
    except Exception as e_scale:
        print(f"\nError scaling G matrix: {e_scale}")
        traceback.print_exc()
        del G_matrix_in_memory; gc.collect(); return None # Modification point

    # --- 7. Complete and return in-memory G matrix ---
    # No longer need del G_memmap and reopening in read-only mode
    print("\nG matrix construction complete (in-memory mode).")
    monitor_matrix_stats(G_matrix_in_memory, f"Final G Matrix ({target_dtype.__name__}, In-Memory Array)") # Modification point

    time_used_g_build = time.time() - start_time_g_build
    print(f"\nTotal G matrix construction time (in-memory mode): {time_used_g_build:.2f} seconds")
    return G_matrix_in_memory # Modification point



# --- A^-1 (Pedigree Inverse Matrix) Component Construction Related Functions ---

@njit
def inbrec(an_idx, ped_0based_idx, f_coeffs, avginb_coeffs):
    """
    [Numba JIT] Recursively calculates the inbreeding coefficient F for a single individual.
    This is part of the Meuwissen and Luo (1992) algorithm, for handling unknown parents.

    Parameters:
    an_idx (int): Position of the individual for whom F is to be calculated, in its 0-based index array.
    ped_0based_idx (np.ndarray): 0-based indexed pedigree data (individual index, sire index, dam index, year group index).
                                   Unknown parents are represented by -1.
    f_coeffs (np.ndarray): Array storing inbreeding coefficients for all individuals for current iteration (or already calculated).
    avginb_coeffs (np.ndarray): Array storing average inbreeding coefficients for each year group.

    Returns:
    float: Inbreeding coefficient F for individual an_idx.
    """
    # Get 0-based indices of parents
    s_idx = ped_0based_idx[an_idx, 1]
    d_idx = ped_0based_idx[an_idx, 2]

    # --- Case 1: At least one parent unknown (index < 0) ---
    if s_idx < 0 or d_idx < 0:
        # For individuals with unknown parents, their inbreeding coefficient F is set to the average
        # inbreeding coefficient of their corresponding year group.
        year_idx = ped_0based_idx[an_idx, 3] # Get year group index
        # Check if year group index is valid and if the average inbreeding coefficient for that group has been calculated (greater than 0)
        if 0 <= year_idx < len(avginb_coeffs) and avginb_coeffs[year_idx] > 0:
             # Use pre-calculated average inbreeding coefficient for the year group
             return avginb_coeffs[year_idx]
        else:
             # If year group index is invalid or average not calculated, return 0.0
             # This usually happens in early iterations or for base generation
             return 0.0

    # --- Case 2: Both parents known (indices >= 0) ---
    else:
        # F_i = 0.5 * a_sd, where a_sd is the additive relationship coefficient between parents
        # Call cffa function to calculate relationship between parents (note cffa returns a_sd)
        # !! Important: cffa calculates 1 + F, here F is needed.
        # F_i = 0.5 * cffa(sire, dam) - This formula seems incorrect, should be F_i = 0.5 * phi_sd
        # Correct formula: F_i = 0.5 * a_sd, and a_sd = cffa(s_idx, d_idx) / 2 ? Incorrect
        # Classical formula: F_i = 0.5 * (phi_ss + phi_dd)/2? Incorrect
        # F_i = 0.5 * a_sd (relationship)? Also incorrect
        # F_i = 0.5 * phi_sd (coancestry coefficient)
        # Refer to Quaas (1976) or Mrode (2014) books:
        # F_x = 0.5 * a_sd (if a_sd is relationship coefficient)
        # Or F_x = phi_sd (if phi_sd is coancestry coefficient)
        # Common recursive formula is based on path coefficients: F_x = sum[(1/2)^n * (1 + F_A)] for paths s->A->d
        # Fast algorithms by Colleau (2002) and Sargolzaei (2005): F_i = 0.5 * CFF(s, d)
        # Where CFF(s,d) is the coancestry coefficient between s and d.

        # Using cffa function (assuming it calculates coancestry coefficient phi_sd)
        # But cffa's implementation looks like it's calculating part of the relationship coefficient a_sd...
        # Let's assume cffa(s,d) * 0.5 is F_i = phi_sd
        # Verification: If s=d, cffa(s,s) = 1 + f_s. F_i = 0.5 * (1+f_s)? Incorrect.
        # If s, d are unrelated and non-inbred, f_s=f_d=0, cffa=0, F_i=0. Correct.
        # If s, d are full siblings, f_s=f_d=0, parents are a, b. cffa(s,d) = 0.5*(cffa(s,a)+cffa(s,b)) = 0.5*(0.5*(cffa(a,a)+cffa(a,b)) + 0.5*(cffa(b,a)+cffa(b,b)))
        #   = 0.25*( (1+f_a) + a_ab + a_ba + (1+f_b) ). If a,b unrelated non-inbred = 0.25*(1+0+0+1)=0.5. F_i = 0.5*0.5 = 0.25. Full sibling F=0.25. Correct.

        # Confirm if cffa implementation correctly calculates phi_sd = a_sd * 0.5?
        # cffa(a, b) seems to calculate a_ab (relationship).
        # F_i = phi_sd = 0.5 * a_sd
        # So F_i = 0.5 * cffa(s_idx, d_idx, ...) ?

        # Re-check cffa implementation:
        # if a1 == a2: return f_coeffs[a1] + 1.0 --> This is a_aa = 1 + F_a
        # if a1 < a2: return 0.5 * (cffa(a1, s_a2) + cffa(a1, d_a2)) --> This is a_xy = 0.5(a_x,s_y + a_x,d_y)
        # So cffa indeed calculates the relationship coefficient a
        # Then F_i = phi_sd = 0.5 * a_sd = 0.5 * cffa(s_idx, d_idx, ...)

        # Ensure parent indices are within valid range of f_coeffs array
        # (In iterative calculation, this usually holds true because parents should be calculated before offspring)
        if not (0 <= s_idx < len(f_coeffs) and 0 <= d_idx < len(f_coeffs)):
             # If parent indices are out of range, may indicate pedigree error or processing order issue
             # Return 0.0 as a safe fallback
             # print(f"Warning: Parent indices ({s_idx}, {d_idx}) for individual {an_idx} are out of f_coeffs range ({len(f_coeffs)}).")
             return 0.0

        # Calculate additive relationship coefficient a_sd between parents
        a_sd = cffa(s_idx, d_idx, ped_0based_idx, f_coeffs, avginb_coeffs)
        # Calculate inbreeding coefficient F_i = 0.5 * a_sd
        # !! Note: According to van Raden (1992) and subsequent algorithms, F_i = phi_sd = a_sd / 2
        #   However, some implementations have F_i = 0.5 * (1 + F_parent_average)? Incorrect
        #   Confirm original source or classic literature. Mrode & Thompson 'Linear Models...' p.81: F_X = phi_SD
        #   And phi_SD = 0.5 * a_SD (coancestry = 0.5 * relationship) - This also seems incorrect, coancestry = sum of path coefficients.
        #   Quaas (1976) F_i = sum[(1/2)^L * (1+F_A)] ?
        #   Henderson (1976) Tabular method: d_ii depends on F_s, F_d. F_i from coancestry.
        #   Let's follow the fast algorithm logic by Colleau(2002)/Sargolzaei(2005): F_i = 0.5 * CFF(s,d)
        #   Assume cffa calculates CFF (Coancestry Factor Function?) or a_sd (relationship)
        #   If cffa is a_sd, then F_i = 0.5 * a_sd.
        #   If cffa is phi_sd, then F_i = phi_sd.
        #   Based on cffa(a,a) = 1+F_a, cffa seems more like a_sd.
        #   So, F_i = 0.5 * a_sd

        # *** Correction ***: Fast algorithms for F usually iterate directly on F itself, not through relationships.
        # F_i = sum((1/2)^L * (1+F_A)) over paths.
        # Another method (which cffa might be trying to implement):
        # L_ii = sqrt(1 - F_i) -> d_ii = L_ii^2 ?
        # F_i can be obtained by iteratively solving diagonal elements of A matrix (A_ii = 1 + F_i).
        # A = T D T', A_ii = sum(T_ik^2 * D_kk).
        # Another direct F calculation method (Meuwissen & Luo, 1992; VanRaden, 1992):
        # F_i = 0.5 * (phi_{s,s} + phi_{d,d}) / 2 ? No.
        # F_i = phi_{s,d} (coancestry coefficient of parents)
        # If using relationship a_{s,d}: phi_{s,d} = a_{s,d} / 2
        # So F_i = a_{s,d} / 2 = cffa(s_idx, d_idx, ...) / 2.0
        # Check cffa again: It recursively calculates a_{xy}.
        # Therefore, the calculation here should be F_i = 0.5 * a_sd
        # However, the code `f_iter[i] = inbrec(...)` and the recursive call of cffa mean that f_coeffs stores F.
        # If cffa(s,d) internally calls f_coeffs[parent] + 1.0, this is incorrect.
        # cffa should only depend on pedigree structure, not on F.

        # *** Re-examine cffa and inbrec logic ***
        # Assume we want to calculate F iteratively using Meuwissen & Luo (1992) method.
        # F_i^(t+1) = function(F_s^t, F_d^t, F_ancestors^t, avg_F_group^t)
        # If parents unknown, F_i = avg_F_group
        # If parents known, F_i = phi_sd (coancestry coefficient)
        # Coancestry coefficient phi_sd can be calculated recursively:
        # phi(x, y) = 0.5 * (phi(x, sire(y)) + phi(x, dam(y)))  (assuming x is 'older' than parents of y)
        # phi(x, x) = 0.5 * (1 + F_x)
        # F_x = phi(sire(x), dam(x))

        # It seems that the `cffa` function's purpose is to calculate `phi(a1, a2)` not `a(a1, a2)`.
        # If `cffa` calculates phi:
        # if a1 == a2: return 0.5 * (1.0 + f_coeffs[a1])  <-- Here f_coeffs is F
        # if a1 < a2: return 0.5 * (cffa(a1, s_a2) + cffa(a1, d_a2))
        # Then F_i = cffa(s_idx, d_idx, ...) in `inbrec` would be correct.

        # *** Modify cffa to calculate coancestry coefficient phi ***
        # (This modification should be done in the cffa function itself, here we assume cffa calculates phi for now)
        # F_i = cffa(s_idx, d_idx, ped_0based_idx, f_coeffs, avginb_coeffs)
        # return F_i

        # *** Keep original code logic, assuming cffa calculates relationship coefficient a ***
        # F_i = 0.5 * a_sd = 0.5 * cffa(...)
        # !!! However, `inbrec` directly returns `0.5 * cffa(...)`, which matches the original call `f_iter[i] = inbrec(...)`.
        # This implies that the original code might assume F_i = 0.5 * a_sd ??? This is consistent with the standard definition F_i = phi_sd = a_sd/2!
        # So the return value of inbrec in the original code is already F_i.

        return 0.5 * cffa(s_idx, d_idx, ped_0based_idx, f_coeffs, avginb_coeffs)

@njit
def cffa(a1_idx, a2_idx, ped_0based_idx, f_coeffs, avginb_coeffs):
    """
    [Numba JIT] Recursively calculates the additive genetic relationship coefficient a (Wright's relationship) between two individuals.
    This is part of the Meuwissen and Luo (1992) algorithm.

    Parameters: (Same as inbrec)
    a1_idx, a2_idx: Positions of the two individuals for whom relationship is to be calculated, in their 0-based index arrays.
    ... (Other parameters same as inbrec)

    Returns:
    float: Additive relationship coefficient a_12 between individuals a1 and a2.
           Note: Returns a_12, not coancestry coefficient phi_12 (phi = a / 2).
           Diagonal elements return a_ii = 1 + F_i.
    """
    # --- Base Cases ---
    # 1. If any individual index is invalid (< 0), indicates unknown individual, relationship is 0
    if a1_idx < 0 or a2_idx < 0:
        return 0.0

    # 2. Calculate self-relationship a_ii = 1 + F_i
    if a1_idx == a2_idx:
        # Ensure index is valid
        if 0 <= a1_idx < len(f_coeffs):
            # Return 1.0 + F_i (F_i from current iteration's f_coeffs array)
            return 1.0 + f_coeffs[a1_idx]
        else:
            # If index is invalid (should theoretically not happen), return 1.0 (assume base non-inbred individual)
            return 1.0

    # --- Recursive Case ---
    # To avoid infinite recursion (e.g., calculating a(i, j) and a(j, i)), ensure recursive calls
    # always proceed towards individuals with smaller indices (usually corresponding to earlier ancestors, if pedigree is sorted).
    # Here, directly compare indices.
    # Ensure indices are within valid range of pedigree array ped_0based_idx
    n_ped_rows = ped_0based_idx.shape[0]
    if not (0 <= a1_idx < n_ped_rows and 0 <= a2_idx < n_ped_rows):
         # print(f"Warning: Indices ({a1_idx}, {a2_idx}) in cffa call are out of pedigree range ({n_ped_rows}).")
         return 0.0 # Invalid index, return 0

    # Assume a1's index is always less than a2's index for recursion (swap if not)
    # This helps utilize caching or avoid redundant computations, and ensures recursion eventually reaches base case
    if a1_idx > a2_idx:
        a1_idx, a2_idx = a2_idx, a1_idx # Swap to make a1_idx <= a2_idx

    # Get a2's parent indices
    s_a2_idx = ped_0based_idx[a2_idx, 1]
    d_a2_idx = ped_0based_idx[a2_idx, 2]

    # Apply recursive formula: a_12 = 0.5 * (a_1,sire(2) + a_1,dam(2))
    term1 = cffa(a1_idx, s_a2_idx, ped_0based_idx, f_coeffs, avginb_coeffs) # a_1,sire(2)
    term2 = cffa(a1_idx, d_a2_idx, ped_0based_idx, f_coeffs, avginb_coeffs) # a_1,dam(2)

    return 0.5 * (term1 + term2)


@njit
def calculate_inbreeding_colleau_modified(ped_data, n_total, max_iterations=1):
    """
    Calculates inbreeding coefficients using a modified Colleau algorithm.

    Parameters:
    ped_data (np.ndarray): Array of shape (n_total, 3), containing [Individual ID, Sire ID, Dam ID]
                         Unknown parents represented by 0 or -1
    n_total (int): Total number of individuals in the population
    max_iterations (int): Maximum number of iterations (usually 1, as this method only needs one iteration)

    Returns:
    np.ndarray: Array containing inbreeding coefficients F for all individuals
    """
    # Initialize inbreeding coefficient array
    f_values = np.zeros(n_total + 1, dtype=np.float64)

    # Extract and re-encode pedigree for all sires and dams
    n_parents = 0
    link = np.zeros(n_total + 1, dtype=np.int32)
    max_id_p = np.zeros(n_total + 1, dtype=np.int32)
    r_ped = np.zeros((n_total + 1, 2), dtype=np.int32)

    # Handle unknown parents
    link[0] = 0

    # Extract and re-encode ancestors
    for i in range(1, n_total + 1):
        sire_id = ped_data[i-1, 1]
        dam_id = ped_data[i-1, 2]

        # Process sire
        if sire_id > 0 and link[sire_id] == 0:
            n_parents += 1
            link[sire_id] = n_parents
            max_id_p[n_parents] = link[sire_id]

            # Get sire's parents
            s_of_sire = ped_data[sire_id-1, 1] if sire_id <= len(ped_data) else 0
            d_of_sire = ped_data[sire_id-1, 2] if sire_id <= len(ped_data) else 0

            r_ped[n_parents, 0] = link[s_of_sire] if s_of_sire > 0 else 0
            r_ped[n_parents, 1] = link[d_of_sire] if d_of_sire > 0 else 0

        # Process dam
        if dam_id > 0 and link[dam_id] == 0:
            n_parents += 1
            link[dam_id] = n_parents

            # Get dam's parents
            s_of_dam = ped_data[dam_id-1, 1] if dam_id <= len(ped_data) else 0
            d_of_dam = ped_data[dam_id-1, 2] if dam_id <= len(ped_data) else 0

            r_ped[n_parents, 0] = link[s_of_dam] if s_of_dam > 0 else 0
            r_ped[n_parents, 1] = link[d_of_dam] if d_of_dam > 0 else 0

        # For each paternal group, determine the maximum ID of parents
        if sire_id > 0 and dam_id > 0:
            if max_id_p[link[sire_id]] < link[dam_id]:
                max_id_p[link[sire_id]] = link[dam_id]

    # Sort animals by sire ID
    sorted_idx = np.zeros(n_total, dtype=np.int32)
    for i in range(n_total):
        sorted_idx[i] = i + 1

    # Simplified sorting: sort by sire ID
    for i in range(n_total):
        for j in range(i+1, n_total):
            if ped_data[sorted_idx[i]-1, 1] > ped_data[sorted_idx[j]-1, 1]:
                # Swap
                temp = sorted_idx[i]
                sorted_idx[i] = sorted_idx[j]
                sorted_idx[j] = temp

    # Calculate inbreeding coefficients
    x = np.zeros(n_total + 1, dtype=np.float64)
    within_family_variance = np.zeros(n_total + 1, dtype=np.float64)

    i = 0
    k = 1  # Index for calculating segregation variance

    while i < n_total:
        animal_idx = sorted_idx[i]
        sire_id = ped_data[animal_idx-1, 1]

        if sire_id <= 0:
            # Sire unknown, inbreeding coefficient is 0
            f_values[animal_idx] = 0.0
            i += 1
            continue

        # Process current sire
        r_sire = link[sire_id]
        mip = max_id_p[r_sire]

        # Initialization
        x[:] = 0.0
        x[r_sire] = 1.0

        # Calculate segregation variance
        while k <= sire_id:
            if link[k] > 0:
                s_k = ped_data[k-1, 1]
                d_k = ped_data[k-1, 2]

                if s_k > 0 and d_k > 0:
                    # Both parents known
                    within_family_variance[link[k]] = 0.5 - 0.25 * (f_values[s_k] + f_values[d_k])
                elif s_k > 0:
                    # Only sire known
                    within_family_variance[link[k]] = 0.75 - 0.25 * f_values[s_k]
                elif d_k > 0:
                    # Only dam known
                    within_family_variance[link[k]] = 0.75 - 0.25 * f_values[d_k]
                else:
                    # Both parents unknown
                    within_family_variance[link[k]] = 1.0
            k += 1

        # Backtrace simplified pedigree
        for j in range(r_sire, 0, -1):
            if x[j] != 0:
                if r_ped[j, 0] > 0:
                    x[r_ped[j, 0]] += x[j] * 0.5
                if r_ped[j, 1] > 0:
                    x[r_ped[j, 1]] += x[j] * 0.5

                x[j] *= within_family_variance[j]

        # Forward trace simplified pedigree
        for j in range(1, mip + 1):
            x[j] += (x[r_ped[j, 0]] + x[r_ped[j, 1]]) * 0.5

        # Calculate inbreeding coefficients for all offspring of current sire
        current_sire_start = i
        while i < n_total and ped_data[sorted_idx[i]-1, 1] == sire_id:
            animal_idx = sorted_idx[i]
            dam_id = ped_data[animal_idx-1, 2]

            if dam_id > 0:
                # Inbreeding coefficient is half of the relationship coefficient between sire and dam
                f_values[animal_idx] = x[link[dam_id]] * 0.5
            else:
                f_values[animal_idx] = 0.0

            i += 1

        # Reset x array for the next sire
        for j in range(1, mip + 1):
            x[j] = 0.0

    return f_values[1:]  # Return inbreeding coefficients array excluding index 0

# Helper function: Convert your pedigree data to required format
def prepare_pedigree_data(ped_data_original):
    """
    Prepares pedigree data for the modified Colleau algorithm.

    Parameters:
    ped_data_original: Original pedigree data, could be in various formats

    Returns:
    np.ndarray: Array of shape (n, 3), containing [Individual ID, Sire ID, Dam ID]
    """
    # Assume ped_data_original is a numpy array of shape (n, 3) or similar structure
    # where columns are [Individual ID, Sire ID, Dam ID]

    # Standardize unknown parents to 0
    ped_data = np.array(ped_data_original, dtype=np.int32)
    ped_data[ped_data < 0] = 0

    return ped_data

def calculate_inbreeding_coefficients(ped_data_internal, final_num_individuals):
    """
    Main function: Wrapper function to calculate inbreeding coefficients.

    Parameters:
    ped_data_internal: Pedigree data, shape (n, 4), containing [Individual Index, Sire Index, Dam Index, Year Group]
    final_num_individuals: Total number of individuals

    Returns:
    np.ndarray: Array of inbreeding coefficients
    """
    start_time = time.time()
    print("Starting inbreeding coefficient calculation (Modified Colleau algorithm)...")

    # Prepare pedigree data
    ped_for_colleau = np.zeros((final_num_individuals, 3), dtype=np.int32)
    for i in range(final_num_individuals):
        ped_for_colleau[i, 0] = i + 1  # Individual ID (1-based)
        ped_for_colleau[i, 1] = ped_data_internal[i, 1] + 1 if ped_data_internal[i, 1] >= 0 else 0  # Sire ID
        ped_for_colleau[i, 2] = ped_data_internal[i, 2] + 1 if ped_data_internal[i, 2] >= 0 else 0  # Dam ID

    # Calculate inbreeding coefficients
    f_coeffs = calculate_inbreeding_colleau_modified(ped_for_colleau, final_num_individuals)

    elapsed_time = time.time() - start_time
    print(f"Inbreeding coefficient calculation complete, time taken: {elapsed_time:.2f} seconds")
    print(f"Inbreeding coefficient statistics: Mean={np.mean(f_coeffs):.4f}, Min={np.min(f_coeffs):.4f}, Max={np.max(f_coeffs):.4f}")

    return f_coeffs

def build_D_inv_from_pedigree(ped_data_np, F_coeffs):
    """
    Constructs the diagonal matrix D^-1 based on pedigree and inbreeding coefficients F.
    [Correction] Corrected d_inv_values initialization and calculation formula.
    """
    print("\n--- Starting D^-1 matrix construction (from pedigree and F coefficients) [Corrected Version] ---")
    n_animals = ped_data_np.shape[0]
    if len(F_coeffs) != n_animals:
        print(f"Error: Inbreeding coefficient F array size ({len(F_coeffs)}) does not match number of individuals ({n_animals}).")
        raise ValueError("F coefficients and pedigree size mismatch")

    # Initialize with 1.0 instead of 0.0, which is a safer default
    d_inv_values = np.ones(n_animals, dtype=np.float64)
    eps = 1e-10 # Small value to prevent division by zero

    print(f"  Calculating {n_animals} diagonal elements of D^-1...")
    for i in tqdm(range(n_animals), desc="Calculating D^-1 diagonal elements", leave=False, ncols=80):
        s = ped_data_np[i, 1] # Sire index
        d = ped_data_np[i, 2] # Dam index
        d_ii_inv = 1.0 # Default value (corresponds to both parents unknown)

        try:
            if s >= 0 and d >= 0: # Both parents known
                if 0 <= s < n_animals and 0 <= d < n_animals:
                    F_s = F_coeffs[s]
                    F_d = F_coeffs[d]
                    # Try using a more standard formula: 1 / (0.5 * (1 - 0.5*(Fs+Fd)))
                    # Note that if Fs+Fd approaches 2, the denominator can be 0
                    denominator = 0.5 * (1.0 - 0.5 * (F_s + F_d))
                    if denominator < eps: # If denominator is close to or less than zero
                        print(f"Warning: Individual {i}'s (parents {s},{d}, Fs={F_s:.3f}, Fd={F_d:.3f}) D^-1 denominator ({denominator:.2e}) is too small or non-positive. Using a large value instead.")
                        d_ii_inv = 1.0 / eps # Set to a very large value
                    else:
                        d_ii_inv = 1.0 / denominator
                else: # Invalid parent indices
                     if i < 10: # Only print first few warnings to avoid flooding
                        print(f"Warning: build_D_inv: Individual {i}'s parent indices ({s}, {d}) are invalid. Using default d_inv=1.0.")
                     d_ii_inv = 1.0
            elif s >= 0 and d < 0: # Only sire known
                if 0 <= s < n_animals:
                    F_s = F_coeffs[s]
                    # Formula: 1 / (1 - 0.25 * a_ss) = 1 / (1 - 0.25 * (1+Fs))
                    denominator = 1.0 - 0.25 * (1.0 + F_s)
                    if denominator < eps:
                         print(f"Warning: Individual {i}'s (sire {s}, Fs={F_s:.3f}) D^-1 denominator ({denominator:.2e}) is too small or non-positive. Using a large value instead.")
                         d_ii_inv = 1.0 / eps
                    else:
                         d_ii_inv = 1.0 / denominator
                else:
                     if i < 10: print(f"Warning: build_D_inv: Individual {i}'s sire index {s} is invalid. Using default d_inv=1.0.")
                     d_ii_inv = 1.0
            elif s < 0 and d >= 0: # Only dam known
                if 0 <= d < n_animals:
                    F_d = F_coeffs[d]
                    denominator = 1.0 - 0.25 * (1.0 + F_d)
                    if denominator < eps:
                         print(f"Warning: Individual {i}'s (dam {d}, Fd={F_d:.3f}) D^-1 denominator ({denominator:.2e}) is too small or non-positive. Using a large value instead.")
                         d_ii_inv = 1.0 / eps
                    else:
                         d_ii_inv = 1.0 / denominator
                else:
                    if i < 10: print(f"Warning: build_D_inv: Individual {i}'s dam index {d} is invalid. Using default d_inv=1.0.")
                    d_ii_inv = 1.0
            # else: Both parents unknown, d_ii_inv remains default 1.0

            d_inv_values[i] = d_ii_inv

        except Exception as e_di:
            print(f"\nError calculating d_inv value for individual {i}: {e_di}")
            d_inv_values[i] = 1.0 # Set to default value on error

    # Construct diagonal sparse matrix D^-1
    D_inv_matrix = sp.diags(d_inv_values, offsets=0, shape=(n_animals, n_animals),
                            format='csr', dtype=np.float64)
    D_inv_matrix.eliminate_zeros() # Remove possible zero values (though theoretically should not exist)

    monitor_matrix_stats(D_inv_matrix, "Constructed D^-1 Matrix [Corrected Version]")

    # --- [Debug] Check reasonableness of D^-1 diagonal values ---
    print("[Debug] D^-1 Diagonal Value Statistics [Corrected Version]:")
    min_d_inv = np.min(d_inv_values); max_d_inv = np.max(d_inv_values); mean_d_inv = np.mean(d_inv_values)
    print(f"  Mean: {mean_d_inv:.4f}")
    print(f"  Min: {min_d_inv:.4f}")
    print(f"  Max: {max_d_inv:.4f}")
    num_lt_one = np.sum(d_inv_values < 1.0 - 1e-6)
    if num_lt_one > 0:
        print(f"  Warning: D^-1 still has {num_lt_one} diagonal values less than 1.0! Please re-check F calculation and formula.")
    # if max_d_inv > 2.0 + 1e-6: # Theoretically d_ii^-1 <= 2 ? (when Fs=Fd=1, denominator is 0.5*(1-0.5*2)=0?)
    #     print(f"  Warning: D^-1 has values greater than 2.0 (max {max_d_inv:.2f}).")
    if np.isinf(max_d_inv) or max_d_inv > 1e6: # Check for very large values
         print(f"  Warning: D^-1 has extremely large values (max {max_d_inv:.2e}), which might indicate a problem.")


    return D_inv_matrix


def build_L_inv_from_pedigree(ped_data_np):
    """
    Constructs the T^-1 = L matrix (lower triangular, with 1s on diagonal) based on pedigree.
    A^-1 = L' D^-1 L, this L is a key component for calculating A^-1.
    L_ii = 1
    L_ij = -0.5 if j is a parent of i
    L_ij = 0 otherwise

    Parameters:
    ped_data_np (np.ndarray): (N, 4) 0-based indexed pedigree data.

    Returns:
    scipy.sparse.csr_matrix: Lower triangular sparse matrix L = T^-1.
    """
    print("\n--- Starting L matrix construction (L = T^-1, from pedigree) ---")
    n_animals = ped_data_np.shape[0]
    if ped_data_np.shape[1] < 3:
        print("Error: Pedigree data has insufficient columns (requires at least individual, sire, dam indices).")
        raise ValueError("Incorrect pedigree data format")

    # Use COO format to efficiently construct sparse matrix, then convert to CSR
    rows, cols, data = [], [], []
    num_invalid_parents = 0

    print(f"  Constructing {n_animals} rows of L matrix...")
    for i in tqdm(range(n_animals), desc="Constructing L Matrix", leave=False, ncols=80): # i is 0-based index of current individual
        s = ped_data_np[i, 1] # Sire index
        d = ped_data_np[i, 2] # Dam index

        # 1. Add diagonal element L_ii = 1.0
        rows.append(i)
        cols.append(i)
        data.append(1.0)

        # 2. Add sire contribution L_is = -0.5 (if sire known and within range)
        if s >= 0: # Sire known
            if 0 <= s < n_animals: # Valid sire index
                rows.append(i)
                cols.append(s)
                data.append(-0.5)
            else: # Invalid sire index
                num_invalid_parents += 1
                # print(f"Warning: build_L_inv: Individual {i}'s sire index {s} is out of range [0, {n_animals-1}]") # May be too verbose

        # 3. Add dam contribution L_id = -0.5 (if dam known and within range)
        if d >= 0: # Dam known
            if 0 <= d < n_animals: # Valid dam index
                rows.append(i)
                cols.append(d)
                data.append(-0.5)
            else: # Invalid dam index
                num_invalid_parents += 1
                # print(f"Warning: build_L_inv: Individual {i}'s dam index {d} is out of range [0, {n_animals-1}]")

    if num_invalid_parents > 0:
        print(f"Warning: Encountered {num_invalid_parents} invalid parent indices while constructing L matrix.")

    # Create matrix using COO format
    L_inv_matrix_coo = coo_matrix((data, (rows, cols)),
                                  shape=(n_animals, n_animals),
                                  dtype=np.float64)

    # Convert to CSR format for better efficiency in subsequent matrix operations
    L_inv_matrix_csr = L_inv_matrix_coo.tocsr()
    # Remove any zero elements that might have been created (e.g., if -0.5 and +0.5 added together, though unlikely here)
    L_inv_matrix_csr.eliminate_zeros()

    monitor_matrix_stats(L_inv_matrix_csr, "Constructed L Matrix (T^-1)")
    return L_inv_matrix_csr

# --- Matrix Verification and Debugging Functions ---

def debug_check_matrix_kernel(matrix_op, name="Matrix", num_tests=5, tol=1e-10):
    """
    [Debug Function] Checks if a matrix has an approximate null space (is close to singular).
    Achieved by calculating the ratio ||Ax|| / ||x|| for random vectors x.
    If the ratio is close to zero, the matrix might be singular or nearly singular.

    Parameters:
    matrix_op (LinearOperator or scipy.sparse matrix): The matrix or linear operator to check.
    name (str): Name of the matrix, for printing information.
    num_tests (int): Number of random vectors to use for testing.
    tol (float): Threshold for determining if the ratio is close to zero.
    """
    if matrix_op is None:
        print(f"[Debug] Warning: {name} matrix is None, skipping null space check.")
        return

    print(f"\n[Debug] Checking approximate null space of {name} matrix:")
    n = matrix_op.shape[0]
    if n == 0:
        print(f"  {name} matrix is empty, skipping check.")
        return

    # Determine matrix-vector multiplication operation
    if isinstance(matrix_op, LinearOperator):
        mat_vec_func = matrix_op.matvec
    elif sp.issparse(matrix_op) or isinstance(matrix_op, np.ndarray):
         mat_vec_func = matrix_op.dot
    else:
         print(f"  Error: Unsupported matrix type {type(matrix_op)} for null space check.")
         return

    np.random.seed(42)  # Fix seed for reproducibility
    min_norm_ratio = np.inf # Initialize minimum ratio to infinity

    print(f"  Testing with {num_tests} random vectors...")
    for i in range(num_tests):
        # Generate random vector and normalize
        x = np.random.randn(matrix_op.shape[1]) # Vector dimension should match matrix's column count
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-15: continue # Skip zero vector
        x = x / x_norm # Input vector has norm 1

        try:
            # Calculate y = Ax
            y = mat_vec_func(x)
            y_norm = np.linalg.norm(y)
            norm_ratio = y_norm # Since ||x|| = 1, ||Ax||/||x|| = ||Ax||

            min_norm_ratio = min(min_norm_ratio, norm_ratio) # Update minimum value
            # print(f"    Test vector {i+1}: ||Ax||/||x|| = {norm_ratio:.4e}")

        except Exception as e:
            print(f"    Calculation failed for test vector {i+1}: {e}")
            min_norm_ratio = 0 # Failure considered as potential problem
            break # If one test fails, stop subsequent tests

    print(f"  Test complete. Minimum norm ratio (min ||Ax||/||x||): {min_norm_ratio:.4e}")
    if min_norm_ratio < tol:
        print(f"  Warning: {name} matrix might be singular or nearly singular (minimum norm ratio < {tol:.1e}).")
    else:
        print(f"  {name} matrix appears numerically stable (minimum norm ratio >= {tol:.1e}).")


def debug_validate_G_matrix(G_mat, check_pd_sample_size=500, sym_check_size=1000, diag_check_size=1000):
    """
    [Debug Function] Validates basic properties of G matrix: symmetry, diagonal elements, (approximate) positive definiteness.

    Parameters:
    G_mat (np.ndarray or np.memmap): G matrix.
    check_pd_sample_size (int): Sample size of sub-matrix used for positive definiteness check.
    sym_check_size (int): Number of element pairs sampled for symmetry check.
    diag_check_size (int): Number of elements sampled for diagonal check.
    """
    if G_mat is None:
        print("[Debug] Error: G matrix is None, cannot validate.")
        return False

    print("\n--- [Debug] G Matrix Property Validation ---")
    is_memmap = isinstance(G_mat, np.memmap)
    g_size = G_mat.shape[0]
    if g_size == 0:
         print("  G matrix is empty, skipping validation.")
         return True
    if G_mat.shape[0] != G_mat.shape[1]:
         print(f"  Error: G matrix is not square ({G_mat.shape}).")
         return False

    print(f"  G Matrix Dimensions: {G_mat.shape}, Type: {'Memmap File' if is_memmap else type(G_mat)}")

    validation_passed = True

    # 1. Symmetry Check (Sampling)
    print("\n  1. Symmetry Check (Sampling)...")
    sym_check_size = min(sym_check_size, g_size * (g_size - 1) // 2 if g_size > 1 else 0) # Adjust upper limit of sample size
    if sym_check_size > 0:
        max_sym_diff = 0.0
        try:
            # Randomly sample non-repeating index pairs (i, j) where i < j
            rows_i = np.random.choice(g_size, sym_check_size, replace=True)
            cols_j = np.random.choice(g_size, sym_check_size, replace=True)
            # Filter out i==j cases
            valid_pairs = rows_i != cols_j
            rows_i = rows_i[valid_pairs]
            cols_j = cols_j[valid_pairs]
            num_valid_pairs = len(rows_i)

            if num_valid_pairs > 0:
                for k in tqdm(range(num_valid_pairs), desc="Checking Symmetry", leave=False, ncols=80):
                    i, j = rows_i[k], cols_j[k]
                    val_ij = G_mat[i, j]
                    val_ji = G_mat[j, i]
                    max_sym_diff = max(max_sym_diff, abs(val_ij - val_ji))

                print(f"    Max symmetry difference based on {num_valid_pairs} random sample pairs: {max_sym_diff:.2e}")
                if max_sym_diff > 1e-6: # Set a tolerance
                    print("    Warning: G matrix might not be symmetric!")
                    validation_passed = False
                else:
                    print("    Symmetry check passed.")
            else:
                 print("    Could not generate valid non-diagonal index pairs for symmetry check.")

        except IndexError:
            print("    Error: Index error during symmetry check.")
            validation_passed = False
        except Exception as e:
            print(f"    Error during symmetry check: {e}")
            validation_passed = False
    else:
         print("    Matrix too small for symmetry sampling check.")

    # 2. Diagonal Element Check (Sampling)
    print("\n  2. Diagonal Element Check (Sampling)...")
    diag_check_size = min(diag_check_size, g_size)
    if diag_check_size > 0:
        try:
            diag_indices = np.random.choice(g_size, diag_check_size, replace=False)
            # Safely read diagonal elements from memmap
            if is_memmap:
                 diag_vals = np.array([G_mat[i,i] for i in diag_indices])
            else: # For in-memory array, can get directly
                 diag_vals = G_mat.diagonal()[diag_indices]

            min_diag = np.min(diag_vals)
            max_diag = np.max(diag_vals)
            mean_diag = np.mean(diag_vals)
            num_nonpos = np.sum(diag_vals <= 1e-10) # Check for non-positive or near-zero

            print(f"    Diagonal statistics based on {diag_check_size} random samples:")
            print(f"      Mean={mean_diag:.4f}, Min={min_diag:.4f}, Max={max_diag:.4f}")
            if num_nonpos > 0:
                print(f"    Warning: G matrix diagonal has {num_nonpos} non-positive/near-zero values!")
                # G matrix diagonal should theoretically be 1+F_i (if built by standard method) or similar value, usually > 0
                validation_passed = False
            else:
                print("    Diagonal element check passed (all positive).")
        except Exception as e:
            print(f"    Diagonal check failed: {e}")
            validation_passed = False
    else:
         print("    Cannot perform diagonal sampling check.")


    # 3. (Approximate) Positive Definiteness Check (via small sample eigenvalues)
    print("\n  3. Approximate Positive Definiteness Check (Small Sample Eigenvalues)...")
    check_pd_sample_size = min(check_pd_sample_size, g_size)
    if check_pd_sample_size > 1: # Requires at least 2x2 matrix
        try:
            print(f"    Extracting {check_pd_sample_size}x{check_pd_sample_size} sub-matrix to calculate eigenvalues...")
            sample_indices = np.random.choice(g_size, check_pd_sample_size, replace=False)
            # Extract sub-matrix G_sample from G (ensure loaded into memory)
            if is_memmap:
                # Less efficient but safe method
                g_sample = np.array([G_mat[i, sample_indices] for i in sample_indices])
            else:
                g_sample = G_mat[np.ix_(sample_indices, sample_indices)]

            # Re-check sub-matrix symmetry, as eigvalsh requires symmetry
            sample_sym_diff = np.max(np.abs(g_sample - g_sample.T))
            if sample_sym_diff > 1e-6:
                 print(f"    Warning: Sampled G sub-matrix is not symmetric (max diff: {sample_sym_diff:.2e}), eigenvalue calculation might be inaccurate.")
                 # Still try to calculate, but results might be unreliable

            try:
                # Calculate eigenvalues for real symmetric matrix
                evals = np.linalg.eigvalsh(g_sample)
                min_eval = np.min(evals)
                max_eval = np.max(evals)
                # Condition number = max eigenvalue / min eigenvalue (absolute value)
                condition_num_sample = abs(max_eval / min_eval) if abs(min_eval) > 1e-12 else np.inf

                print(f"    Min eigenvalue of G sub-sample: {min_eval:.4e}")
                print(f"    Max eigenvalue of G sub-sample: {max_eval:.4e}")
                print(f"    Condition number of G sub-sample (estimate): {condition_num_sample:.4e}")

                # Check for negative or near-zero eigenvalues
                tol = 1e-8 # Set tolerance
                num_nonpos_evals = np.sum(evals <= tol)
                if num_nonpos_evals > 0:
                    print(f"    Warning: G sub-sample found {num_nonpos_evals} non-positive or near-zero ({tol:.1e}) eigenvalues. Minimum value: {min_eval:.4e}")
                    validation_passed = False
                    if min_eval < -tol: # If significantly negative
                         fix_add = abs(min_eval) + 1e-6 # Suggested value to add
                         print(f"      Suggestion: Add at least {fix_add:.2e} to the diagonal of G matrix to try to improve positive definiteness.")
                else:
                     print("    G sub-sample eigenvalues are all greater than zero, passed approximate positive definiteness check.")

                # Check if condition number is too large
                if condition_num_sample > 1e8: # Empirical threshold
                    print(f"    Warning: G sub-sample condition number ({condition_num_sample:.2e}) is too large, might lead to numerical issues during MME solving.")
                    # Does not necessarily mean validation failed, but something to note

            except np.linalg.LinAlgError as e_linalg:
                print(f"    Eigenvalue calculation failed (LinAlgError): {e_linalg}")
                validation_passed = False
            except Exception as e_eig:
                print(f"    Unknown error during eigenvalue calculation: {e_eig}")
                validation_passed = False

        except MemoryError:
             print(f"    Insufficient memory to extract {check_pd_sample_size}x{check_pd_sample_size} sub-matrix for eigenvalue check.")
             # Cannot perform this check, but not marked as failure
        except Exception as e_sample:
            print(f"    Failed to extract G sub-sample or calculate eigenvalues: {e_sample}")
            validation_passed = False
    else:
         print("    Matrix too small or sample size insufficient for eigenvalue check.")

    print("\n--- G Matrix Property Validation Ended ---")
    if validation_passed:
         print("  G matrix passed all basic property checks.")
    else:
         print("  Warning: G matrix failed some basic property checks. Please carefully examine construction process and results.")

    return validation_passed

# -----------------------------------------------------------------------------
# 6. Core Solver and Result Processing
# -----------------------------------------------------------------------------

class CGCallback:
    """
    Enhanced Conjugate Gradient (CG) callback function, used for monitoring convergence process and saving intermediate states.
    """
    def __init__(self, system_matrix_cb, rhs_vector_cb, state_file_cb,
                 start_iter_cb=0, total_max_iter_cb=1000, save_freq_cb=100, rtol=1e-8, debug=True):
        """
        Initializes the callback object.

        Parameters:
        system_matrix_cb (LinearOperator): Coefficient matrix (S) of the linear system being solved.
        rhs_vector_cb (np.ndarray): Right-hand side vector (b) of the linear system.
        state_file_cb (str): File path for saving iteration state.
        start_iter_cb (int): Starting iteration count (for resuming from interruption).
        total_max_iter_cb (int): Maximum total iterations for CG solver.
        save_freq_cb (int): How often to save state (every N iterations).
        rtol (float): Relative convergence tolerance (for printing and logging).
        debug (bool): Whether to print detailed convergence information.
        """
        self.system_matrix = system_matrix_cb
        self.rhs_vector = rhs_vector_cb
        self.rhs_norm = np.linalg.norm(rhs_vector_cb) # Calculate norm of RHS vector, for relative residual calculation
        self.state_file = state_file_cb
        self.iteration = start_iter_cb # Global iteration counter (starts from loaded state)
        self.total_max_iterations = total_max_iter_cb
        self.save_frequency = save_freq_cb
        self.cg_call_iter_count = 0 # Iteration counter inside current CG call
        self.relative_tolerance = rtol # Store relative tolerance
        self.debug_print = debug

        # Convergence monitoring related variables
        self.residuals = [] # Store history of absolute residual norms
        self.rel_residuals = [] # Store history of relative residual norms
        self.timestamps = [] # Store history of timestamps
        self.start_time = time.time() # Record callback start time
        self.last_print_time = self.start_time
        self.min_rel_residual = np.inf # Record minimum relative residual encountered
        self.stagnation_counter = 0 # Counter for detecting convergence stagnation
        self.stagnation_threshold = 5 * save_freq_cb # Warn after this many iterations of stagnation

        # Calculate initial relative residual (if starting from 0)
        if start_iter_cb == 0:
            try:
                initial_residual_norm = self.rhs_norm # ||b - A*0|| = ||b||
                initial_rel_residual = 1.0 if self.rhs_norm > 1e-15 else 0.0
                self.residuals.append(initial_residual_norm)
                self.rel_residuals.append(initial_rel_residual)
                self.timestamps.append(0.0)
                self.min_rel_residual = initial_rel_residual
                if self.debug_print:
                    print(f"CG Initial Relative Residual: {initial_rel_residual:.4e} (Target < {self.relative_tolerance:.1e})")
            except Exception as e:
                print(f"Error calculating initial residual: {e}")

    def __call__(self, xk_current_solution):
        """
        Function called at each CG iteration.
        xk_current_solution is the current iteration's solution vector.
        """
        self.iteration += 1 # Update global iteration count
        self.cg_call_iter_count += 1 # Update iteration count for current call
        current_time = time.time() - self.start_time

        # --- Calculate current residual (not needed every time, can reduce frequency for performance) ---
        # Set print and save frequency
        # Dynamically adjust print interval, frequent at start, sparse later
        print_interval = min(max(10, self.cg_call_iter_count // 20), 100)
        should_calculate_residual = (self.cg_call_iter_count % print_interval == 1) or (self.cg_call_iter_count <= 10) # More calculations at start
        should_save_state = (self.iteration % self.save_frequency == 0)

        if should_calculate_residual or should_save_state:
            try:
                # Calculate residual vector r = b - A*xk
                current_residual_vec = self.rhs_vector - self.system_matrix @ xk_current_solution
                current_residual_norm = np.linalg.norm(current_residual_vec)
                # Calculate relative residual ||r|| / ||b|| (add epsilon to prevent division by zero)
                current_rel_residual = current_residual_norm / (self.rhs_norm + 1e-15)

                # Record residual and timestamp
                self.residuals.append(current_residual_norm)
                self.rel_residuals.append(current_rel_residual)
                self.timestamps.append(current_time)

                # --- Check for convergence stagnation ---
                if current_rel_residual < self.min_rel_residual * 0.9999: # If significant improvement
                    self.min_rel_residual = current_rel_residual
                    self.stagnation_counter = 0 # Reset stagnation counter
                else:
                    # If improvement is not significant, increment stagnation counter
                    # The increment equals the check interval, meaning no improvement during this period
                    self.stagnation_counter += print_interval

                # --- Print progress information ---
                if self.debug_print and should_calculate_residual:
                    # Estimate convergence rate (current residual / last recorded residual)
                    conv_rate_str = ""
                    if len(self.rel_residuals) >= 2 and self.rel_residuals[-2] > 1e-15:
                         # Use the last *calculated* residual, not an estimate from the last iteration
                         conv_factor = (current_rel_residual / self.rel_residuals[-2])**(1.0/print_interval) if print_interval > 0 else 1.0
                         conv_rate_str = f", Rate≈{conv_factor:.4f}"

                    print(f"CG Iter {self.iteration:>5}/{self.total_max_iterations} "
                          f"(Current: {self.cg_call_iter_count:>5}) | RelRes={current_rel_residual:.4e}{conv_rate_str} | Time={current_time:.1f}s")

                    # If no significant improvement for a long time, issue a warning
                    if self.stagnation_counter >= self.stagnation_threshold and self.cg_call_iter_count > 200: # Only check in later iterations
                         print(f"  Warning: CG might be stagnating, relative residual has not significantly improved over the last approx. {self.stagnation_counter} iterations.")
                         print(f"        Current minimum relative residual: {self.min_rel_residual:.4e}")
                         self.stagnation_counter = 0 # Reset counter to avoid repetitive warnings

                # --- Save state ---
                if should_save_state:
                    if self.debug_print:
                         print(f"  Saving state (iteration {self.iteration}), current relative residual: {current_rel_residual:.4e}")
                    save_state(self.iteration, xk_current_solution, current_rel_residual, self.state_file)

                    # [Debug] Briefly check solution vector
                    if self.debug_print:
                        xk_norm = np.linalg.norm(xk_current_solution)
                        if np.any(np.isnan(xk_current_solution)) or np.any(np.isinf(xk_current_solution)):
                            print("  Warning: Current solution vector contains NaN or Inf!")
                        elif xk_norm < 1e-15:
                             print("  Warning: Current solution vector norm is close to zero.")


            except Exception as e:
                print(f"\nCallback function error at iteration {self.iteration}: {e}")
                traceback.print_exc()
                # Can choose to raise exception here to stop CG, or just print error and continue
                # raise e # Stop CG

# --- Implicit Matrix-Vector Product Functions for A^-1 and A_22_ped^-1 ---
# (These two functions were defined in previous code blocks, assuming they are available here)
# def apply_A_inv_implicit(v_vec, L_inv_op, D_inv2_op, debug=False): ...
# def apply_A_22_ped_inv_implicit_op(v_ag, L_inv_ag_op, D_inv2_ag_op, debug=False): ...

# --- Implicit Matrix-Vector Product Functions for A^-1 and A_22_ped^-1 ---
# (These functions are now defined in global scope for the preprocessor to call)

def apply_A_inv_implicit(v_vec, L_inv_op, D_inv_op, debug=False):
    """
    [Global Function] Applies A^-1 = L' D^-1 L matrix transformation (A is the pedigree relationship matrix).

    Parameters:
    v_vec (np.ndarray): Input vector.
    L_inv_op (scipy.sparse matrix): L = T^-1 matrix.
    D_inv_op (scipy.sparse matrix): D^-1 diagonal matrix.
    debug (bool): Whether to print debug information.

    Returns:
    np.ndarray: Result of A^-1 @ v_vec.
    """
    v_vec_arr = np.asarray(v_vec, dtype=np.float64)
    if debug: print(f"  apply_A_inv_implicit input norm: {np.linalg.norm(v_vec_arr):.4e}")

    # step1 = L @ v
    step1 = L_inv_op.dot(v_vec_arr)
    if debug: print(f"    L @ v norm: {np.linalg.norm(step1):.4e}")

    # step2 = D^-1 @ step1
    step2 = D_inv_op.dot(step1) # D_inv_op is diagonal matrix, equivalent to element-wise multiplication
    if debug: print(f"    D^-1 @ L @ v norm: {np.linalg.norm(step2):.4e}")

    # result = L' @ step2
    result = L_inv_op.T.dot(step2)
    if debug: print(f"  apply_A_inv_implicit output norm: {np.linalg.norm(result):.4e}")
    return result

def apply_A_inv_block_implicit_op(v_block_op, L_inv_op, D_inv_op, n1_op, n2_op, block_type_op, debug=False):
    """
    [Global Function] Applies block operation of A^-1. Used in ssGBLUP to calculate (A^-1)_11, (A^-1)_12, etc.

    Parameters:
        v_block_op: Input vector (corresponding block).
        L_inv_op, D_inv_op: Components of full A^-1.
        n1_op, n2_op: Number of non-genotyped and genotyped individuals.
        block_type_op: Type of block to calculate ('11', '12', '21', '22').
        debug: Whether to print debug information.

    Returns:
        np.ndarray: Result of the corresponding block.
    """
    v_block_op_arr = np.asarray(v_block_op, dtype=np.float64)
    if debug: print(f"  apply_A_inv_block (block {block_type_op}) input norm: {np.linalg.norm(v_block_op_arr):.4e}")

    # Construct full input vector (pad with zeros)
    v_full_op = np.zeros(n1_op + n2_op, dtype=np.float64)
    if block_type_op in ['11', '21']: # Input corresponds to first part (n1)
        if len(v_block_op_arr) != n1_op: raise ValueError(f"Block {block_type_op} input size error")
        if n1_op > 0: v_full_op[:n1_op] = v_block_op_arr
    elif block_type_op in ['12', '22']: # Input corresponds to second part (n2)
        if len(v_block_op_arr) != n2_op: raise ValueError(f"Block {block_type_op} input size error")
        if n2_op > 0: v_full_op[n1_op:] = v_block_op_arr
    else: raise ValueError(f"Invalid block type: {block_type_op}")

    # Apply full A^-1
    A_inv_v_full_op = apply_A_inv_implicit(v_full_op, L_inv_op, D_inv_op, debug=False) # Reduce debug info in sub-call

    # Extract corresponding result block
    result = None
    if block_type_op in ['11', '12']: # Output corresponds to first part (n1)
        result = A_inv_v_full_op[:n1_op] if n1_op > 0 else np.array([], dtype=np.float64)
    elif block_type_op in ['21', '22']: # Output corresponds to second part (n2)
        result = A_inv_v_full_op[n1_op:] if n2_op > 0 else np.array([], dtype=np.float64)

    if debug: print(f"  apply_A_inv_block (block {block_type_op}) output norm: {np.linalg.norm(result):.4e}")
    return result


def apply_A_22_ped_inv_implicit_op(v_ag, L_inv_ag_op, D_inv_ag_op, debug=False):
    """
    [Global Function] Applies implicit operation of A_22_ped^-1 (only for ssGBLUP).
    A_22_ped^-1 = L_ag' D_ag^-1 L_ag

    Parameters:
    v_ag (np.ndarray): Input vector (length is number of genotyped individuals n_gen).
    L_inv_ag_op (scipy.sparse matrix): L_ag matrix for the genotyped subset.
    D_inv_ag_op (scipy.sparse matrix): D_ag^-1 matrix for the genotyped subset.
    debug (bool): Whether to print debug information.

    Returns:
    np.ndarray: Result of A_22_ped^-1 @ v_ag.
    """
    v_ag_arr = np.asarray(v_ag, dtype=np.float64)
    if L_inv_ag_op is None or D_inv_ag_op is None:
         raise ValueError("apply_A_22_ped_inv requires L_inv_ag and D_inv_ag matrices.")

    if debug: print(f"  apply_A_22_ped_inv input norm: {np.linalg.norm(v_ag_arr):.4e}")

    step1 = L_inv_ag_op.dot(v_ag_arr)
    if debug: print(f"    L_ag @ v norm: {np.linalg.norm(step1):.4e}")

    step2 = D_inv_ag_op.dot(step1)
    if debug: print(f"    D_ag^-1 @ L_ag @ v norm: {np.linalg.norm(step2):.4e}")

    result = L_inv_ag_op.T.dot(step2)
    if debug: print(f"  apply_A_22_ped_inv output norm: {np.linalg.norm(result):.4e}")
    return result
# --- Main Solver Function ---
def build_and_solve_linear_system(
    model_type,
    X_solve, Y_solve, Z_solve, k_solve,
    L_inv_full=None, D_inv_full=None,
    G_solve=None,
    L_inv_ag=None, D_inv_ag=None,
    n_nongen=0, n_gen=0,
    diag_A_inv_vector=None, # <-- Confirm this parameter is added
    state_file_solve='iteration_state.pkl',
    max_iterations_solve=10000,
    atol_solve=1e-10,
    rtol_solve=1e-8,
    preconditioner_omega=0.7,
    debug_solver=True,
    **solver_kwargs # <-- Confirm this is the last parameter
    ):

    """
    Constructs and iteratively solves the Mixed Model Equations (MME).
    Supports ABLUP, GBLUP, and ssGBLUP based on model_type.

    Parameters:
    model_type (str): Model type ('ABLUP', 'GBLUP', 'ssGBLUP').
    X_solve (np.ndarray): Fixed effects design matrix (n_obs x n_fixed).
    Y_solve (np.ndarray): Observation vector (n_obs).
    Z_solve (scipy.sparse.csr_matrix): Random effects design matrix (n_obs x n_random).
                                        For GBLUP, n_random only includes genotyped individuals.
                                        For ABLUP/ssGBLUP, n_random is total individuals.
    k_solve (float): Variance ratio lambda = sigma_e^2 / sigma_u^2.
    L_inv_full (scipy.sparse.csr_matrix, optional): L matrix (T^-1) for ABLUP/ssGBLUP.
    D_inv_full (scipy.sparse.csr_matrix, optional): D^-1 matrix for ABLUP/ssGBLUP.
    G_solve (np.ndarray or np.memmap, optional): G matrix for GBLUP/ssGBLUP.
    L_inv_ag (scipy.sparse.csr_matrix, optional): L_ag component for A_22_ped^-1 in ssGBLUP.
    D_inv_ag (scipy.sparse.csr_matrix, optional): D_ag^-1 component for A_22_ped^-1 in ssGBLUP.
    n_nongen (int): Number of non-genotyped individuals in ssGBLUP.
    n_gen (int): Number of genotyped individuals in ssGBLUP/GBLUP.
    state_file_solve (str): Iteration state save file.
    max_iterations_solve (int): Maximum iterations for CG.
    atol_solve (float): Absolute tolerance for CG.
    rtol_solve (float): Relative tolerance for CG.
    preconditioner_omega (float): Omega parameter for block diagonal preconditioner.
    debug_solver (bool): Whether to print detailed debug information inside the solver.
    solver_kwargs (dict): Additional parameters passed to callback or debug functions.

    Returns:
    tuple: (x_final_solution, gamma_solution, solve_info)
           x_final_solution: Recovered original solution [beta_hat, u_hat].
           gamma_solution: Solution gamma directly from CG solver.
           solve_info: Exit information from CG solver (0 indicates success).
           If failed, may return (None, None, info) or (last_good_solution, last_good_gamma, info).
    """

    print(f"\n--- Starting construction and solving of linear system (Model: {model_type}, k={k_solve:.4f}) ---")
    solve_overall_start_time = time.time()

    # --- Input validation and dimension determination ---
    n_obs, n_fixed = X_solve.shape
    n_random = Z_solve.shape[1] # Total number of random effects

    print(f"System Dimensions: Observations={n_obs}, Fixed Effects={n_fixed}, Random Effects={n_random}")

    # Check required inputs based on model type
    if model_type == 'ABLUP':
        if L_inv_full is None or D_inv_full is None:
            raise ValueError("ABLUP requires L_inv_full and D_inv_full matrices.")
        if L_inv_full.shape[0] != n_random or D_inv_full.shape[0] != n_random:
            raise ValueError(f"ABLUP's L_inv/D_inv dimensions ({L_inv_full.shape[0]}) do not match Z matrix columns ({n_random}).")
        print("Model Type: ABLUP (Pedigree-based)")
    elif model_type == 'GBLUP':
        if G_solve is None:
            raise ValueError("GBLUP requires G_solve matrix.")
        # Assume Z_solve columns n_random now equals number of genotyped individuals n_gen
        n_gen = n_random
        n_nongen = 0
        if G_solve.shape[0] != n_gen:
             raise ValueError(f"GBLUP's G matrix dimension ({G_solve.shape[0]}) does not match Z matrix columns ({n_random}).")
        print(f"Model Type: GBLUP (Genomic-based, Individuals={n_gen})")
    elif model_type == 'ssGBLUP':
        if L_inv_full is None or D_inv_full is None or G_solve is None or L_inv_ag is None or D_inv_ag is None:
            raise ValueError("ssGBLUP requires L_inv_full, D_inv_full, G_solve, L_inv_ag, D_inv_ag.")
        if n_gen <= 0: raise ValueError("ssGBLUP requires genotyped individuals n_gen > 0.")
        if n_nongen + n_gen != n_random:
            raise ValueError(f"ssGBLUP counts mismatch: n_nongen({n_nongen}) + n_gen({n_gen}) != n_random({n_random}).")
        if G_solve.shape[0] != n_gen or L_inv_ag.shape[0] != n_gen or D_inv_ag.shape[0] != n_gen:
             raise ValueError("ssGBLUP's G/L_inv_ag/D_inv_ag dimensions do not match n_gen.")
        if L_inv_full.shape[0] != n_random or D_inv_full.shape[0] != n_random:
             raise ValueError("ssGBLUP's L_inv_full/D_inv_full dimensions do not match n_random.")
        print(f"Model Type: ssGBLUP (Single-step, Non-genotyped={n_nongen}, Genotyped={n_gen})")
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Please choose 'ABLUP', 'GBLUP', or 'ssGBLUP'.")

    # Ensure inputs are float64
    X_s = np.asarray(X_solve, dtype=np.float64)
    Y_s = np.asarray(Y_solve, dtype=np.float64)
    Z_s = Z_solve.astype(np.float64) if Z_solve.dtype != np.float64 else Z_solve

    # Define implicit matrix application functions (assuming they are in scope or imported)
    # apply_A_inv_implicit, apply_A_22_ped_inv_implicit_op, apply_A_inv_block_implicit_op

    # --- Construct Linear Operator for MME Coefficient Matrix S ---
    # S = [ S11 S12 ] = [ X'X   X'Z T ]
    #     [ S21 S22 ]   [ T'Z'X T'(Z'Z+k*H^-1)T ]  (ssGBLUP case, transformed)
    # Or simplified forms for ABLUP/GBLUP

    print("Constructing MME Linear Operator...")
    # --- S11 = X'X (Same for all models) ---
    def S11_matvec(v_beta):
        return X_s.T @ (X_s @ v_beta)
    S11_op = LinearOperator((n_fixed, n_fixed), matvec=S11_matvec, dtype=np.float64)
    monitor_matrix_stats(S11_op, "S11 (X'X) Operator")

    # --- Define matvec functions for S12, S21, S22 (based on model type) ---
    S12_matvec = None
    S21_matvec = None
    S22_matvec = None
    transform_T = None # Transformation matrix T (LinearOperator or None)
    rhs_transform_T_prime = None # For RHS vector transformation T' (LinearOperator or None)

    if model_type == 'ABLUP':
        # ABLUP MME: [ X'X   X'Z         ] [ beta ] = [ X'y ]
        #              [ Z'X   Z'Z + kA^-1 ] [ u    ]   [ Z'y ]
        # No transformation T needed (T=I), gamma = [beta, u]
        # RHS = [X'y, Z'y]

        def S12_ablup_matvec(v_u): # S12 = X'Z
            return X_s.T @ (Z_s @ v_u)
        S12_matvec = S12_ablup_matvec

        def S21_ablup_matvec(v_beta): # S21 = Z'X
            return Z_s.T @ (X_s @ v_beta)
        S21_matvec = S21_ablup_matvec

        def S22_ablup_matvec(v_u): # S22 = Z'Z + k * A^-1
            term1 = Z_s.T @ (Z_s @ v_u)
            term2 = k_solve * apply_A_inv_implicit(v_u, L_inv_full, D_inv_full, debug=debug_solver)
            return term1 + term2
        S22_matvec = S22_ablup_matvec

        # In ABLUP, T is identity matrix, gamma is the original solution [beta, u]
        transform_T = None # Or identity(n_fixed + n_random)? None is simpler
        rhs_transform_T_prime = None

    elif model_type == 'GBLUP':
        # GBLUP MME (genotyped individuals only): [ X'X   X'Zg        ] [ beta ] = [ X'y ]
        #                         [ Zg'X  Zg'Zg + kG^-1 ] [ ug   ]   [ Zg'y]
        # Use transformation T = diag(I, G) to avoid G^-1
        # Transformed system S * gamma = T' * b
        # gamma = [beta, G^-1 * ug]
        # S = [ X'X      X'ZgG        ]
        #     [ GZg'X    GZg'ZgG + kG ]
        # T' * b = [X'y, GZg'y]

        if G_solve is None: raise ValueError("GBLUP requires G_solve matrix.")
        # Z_s should now only contain columns for genotyped individuals Zg

        def S12_gblup_matvec(v_gamma_u): # S12 = X' Zg G
            # v_gamma_u is the u part of gamma, i.e., G^-1 ug
            # We need to calculate X'Zg * (G * v_gamma_u)
            # (Or directly assume input is gamma_u?) CG input is gamma
            return X_s.T @ (Z_s @ (G_solve @ v_gamma_u))
        S12_matvec = S12_gblup_matvec

        def S21_gblup_matvec(v_beta): # S21 = G Zg' X
            return G_solve @ (Z_s.T @ (X_s @ v_beta))
        S21_matvec = S21_gblup_matvec

        def S22_gblup_matvec(v_gamma_u): # S22 = G Zg' Zg G + k G
            # Calculate G * v_gamma_u
            G_gamma_u = G_solve @ v_gamma_u
            # Calculate Zg * G_gamma_u
            Zg_G_gamma_u = Z_s @ G_gamma_u
            # Calculate Zg' * Zg_G_gamma_u
            term1_part1 = Z_s.T @ Zg_G_gamma_u
            # Calculate G * term1_part1
            term1 = G_solve @ term1_part1
            # Calculate k * G * v_gamma_u
            term2 = k_solve * G_gamma_u # Reuse G_gamma_u
            return term1 + term2
        S22_matvec = S22_gblup_matvec

        # Define transformation operator T = diag(I, G)
        def T_gblup_matvec(v):
             res = np.zeros_like(v)
             res[:n_fixed] = v[:n_fixed] # I * beta
             res[n_fixed:] = G_solve @ v[n_fixed:] # G * u_gamma
             return res
        transform_T = LinearOperator((n_fixed+n_gen, n_fixed+n_gen), matvec=T_gblup_matvec, dtype=np.float64)

        # Define T' operator (since G is symmetric, T'=T)
        def T_prime_gblup_matvec(v):
            res = np.zeros_like(v)
            res[:n_fixed] = v[:n_fixed] # I * b_beta
            res[n_fixed:] = G_solve @ v[n_fixed:] # G * b_u
            return res
        rhs_transform_T_prime = LinearOperator((n_fixed+n_gen, n_fixed+n_gen), matvec=T_prime_gblup_matvec, dtype=np.float64)


    elif model_type == 'ssGBLUP':
        # ssGBLUP uses transformation T = diag(I, block_diag(I_nn, G))
        # S and RHS are calculated as per original code and theoretical documentation
        if G_solve is None or L_inv_full is None or D_inv_full is None or L_inv_ag is None or D_inv_ag is None:
             raise ValueError("ssGBLUP missing required matrix components.")

        # Split Z matrix
        Z1_s = Z_s[:, :n_nongen].tocsr() if n_nongen > 0 else None # Non-genotyped part
        Z2_s = Z_s[:, n_nongen:].tocsr() if n_gen > 0 else None    # Genotyped part

        def S12_ssgblup_matvec(v_gamma_u): # S12 = X' Z T
            # v_gamma_u = [gamma_u1, gamma_u2] = [u1, G^-1 u2]
            gamma_u1 = v_gamma_u[:n_nongen] if n_nongen > 0 else None
            gamma_u2 = v_gamma_u[n_nongen:] if n_gen > 0 else None
            # T * gamma_u = [u1, G * G^-1 u2] = [u1, u2] = u
            # But S12 itself is X' Z T, so need to calculate X' Z (T v_gamma_u)
            # T v_gamma_u = [gamma_u1, G gamma_u2]
            term1 = X_s.T @ (Z1_s @ gamma_u1) if n_nongen > 0 and Z1_s is not None else 0.0
            term2 = X_s.T @ (Z2_s @ (G_solve @ gamma_u2)) if n_gen > 0 and Z2_s is not None else 0.0
            return term1 + term2
        S12_matvec = S12_ssgblup_matvec

        def S21_ssgblup_matvec(v_beta): # S21 = T' Z' X
            # T' = T because G is symmetric
            # T' (Z' X v_beta) = [ (Z1'X v_beta), G (Z2'X v_beta) ]
            res_parts = []
            ZTX_beta = Z_s.T @ (X_s @ v_beta)
            if n_nongen > 0:
                res_parts.append(ZTX_beta[:n_nongen])
            if n_gen > 0:
                res_parts.append(G_solve @ ZTX_beta[n_nongen:])
            return np.concatenate(res_parts) if res_parts else np.array([], dtype=np.float64)
        S21_matvec = S21_ssgblup_matvec

        # S22 uses previous S22_matvec_with_debug implementation
        # Need to ensure it's available in current scope or redefine
        def S22_ssgblup_matvec_with_debug(v_gamma_u, debug_every=500):
            """ssGBLUP S22 matrix-vector multiplication with detailed debug information"""
            # ... (internal implementation omitted, same as previous version) ...
            # Calls helper functions and G_solve, L_inv_full, D_inv_full, L_inv_ag, D_inv_ag etc.
            static_count = getattr(S22_ssgblup_matvec_with_debug, 'count', 0)
            S22_ssgblup_matvec_with_debug.count = static_count + 1
            debug = debug_solver and ((static_count % debug_every == 0) or (static_count < 5))

            if debug: print(f"\nS22_ssgblup_matvec call #{static_count}:")

            gamma_u1 = v_gamma_u[:n_nongen] if n_nongen > 0 else None
            gamma_u2 = v_gamma_u[n_nongen:] if n_gen > 0 else None
            if debug:
                 print(f"  Input gamma_u: Total Norm={np.linalg.norm(v_gamma_u):.4e}")
                 if gamma_u1 is not None: print(f"    gamma_u1 Norm: {np.linalg.norm(gamma_u1):.4e}")
                 if gamma_u2 is not None: print(f"    gamma_u2 Norm: {np.linalg.norm(gamma_u2):.4e}")


            res_top = np.zeros(n_nongen, dtype=np.float64) if n_nongen > 0 else np.array([],dtype=np.float64)
            res_bot = np.zeros(n_gen, dtype=np.float64) if n_gen > 0 else np.array([],dtype=np.float64)

            # --- Calculate T'(Z'Z)T * gamma_u ---
            # T * gamma_u = [gamma_u1, G*gamma_u2] = u_hat (approx)
            u_hat_approx = np.concatenate([gamma_u1 if gamma_u1 is not None else [],
                                           G_solve @ gamma_u2 if gamma_u2 is not None else []])
            # Z * u_hat_approx
            Zu_hat = Z_s @ u_hat_approx
            # Z' * Zu_hat
            ZTZu_hat = Z_s.T @ Zu_hat
            # T' * ZTZu_hat = [ (ZTZu_hat)_1, G * (ZTZu_hat)_2 ]
            term1_top = ZTZu_hat[:n_nongen] if n_nongen > 0 else []
            term1_bot = G_solve @ ZTZu_hat[n_nongen:] if n_gen > 0 else []

            if n_nongen > 0: res_top += term1_top
            if n_gen > 0: res_bot += term1_bot
            if debug:
                 if n_nongen > 0: print(f"  S22 - T'Z'ZT part (top): Norm={np.linalg.norm(term1_top):.4e}")
                 if n_gen > 0: print(f"  S22 - T'Z'ZT part (bot): Norm={np.linalg.norm(term1_bot):.4e}")

            # --- Calculate T'(k*H^-1)T * gamma_u ---
            # H^-1 = A^-1 + diag(0, G^-1 - A_22^-1)
            # T'(k*A^-1)T * gamma_u = k * T' A^-1 T gamma_u
            # T'(k*diag(0, G^-1 - A_22^-1))T * gamma_u

            # 1) Calculate k * T' A^-1 T gamma_u
            T_gamma_u = u_hat_approx # T*gamma_u
            A_inv_T_gamma_u = apply_A_inv_implicit(T_gamma_u, L_inv_full, D_inv_full)
            T_prime_A_inv_T_gamma_u = np.concatenate([
                A_inv_T_gamma_u[:n_nongen] if n_nongen > 0 else [],
                G_solve @ A_inv_T_gamma_u[n_nongen:] if n_gen > 0 else []
            ])
            term2 = k_solve * T_prime_A_inv_T_gamma_u
            if n_nongen > 0: res_top += term2[:n_nongen]
            if n_gen > 0: res_bot += term2[n_nongen:]
            if debug:
                 if n_nongen > 0: print(f"  S22 - kT'A^-1T part (top): Norm={np.linalg.norm(term2[:n_nongen]):.4e}")
                 if n_gen > 0: print(f"  S22 - kT'A^-1T part (bot): Norm={np.linalg.norm(term2[n_nongen:]):.4e}")


            # 2) Calculate k * T' diag(0, G^-1 - A_22^-1) T gamma_u
            # T' diag(0, M) T = diag(0, G M G) where M = G^-1 - A_22^-1
            if n_gen > 0 and gamma_u2 is not None:
                # Target vector is gamma_u = [gamma_u1, gamma_u2]
                # We need M * (T*gamma_u)_2 = M * (G * gamma_u2)
                G_gamma_u2 = G_solve @ gamma_u2
                # M * G_gamma_u2 = (G^-1 - A_22^-1) * G_gamma_u2
                #  = G^-1 * G_gamma_u2 - A_22^-1 * G_gamma_u2
                #  = gamma_u2 - apply_A_22_ped_inv_implicit(G_gamma_u2, L_inv_ag, D_inv_ag)
                term_M_G_gamma_u2 = gamma_u2 - apply_A_22_ped_inv_implicit_op(G_gamma_u2, L_inv_ag, D_inv_ag)

                # Now calculate G * term_M_G_gamma_u2
                term3_bot = G_solve @ term_M_G_gamma_u2
                res_bot += k_solve * term3_bot
                if debug: print(f"  S22 - kT'diag(0,Ginv-A22inv)T part (bot): Norm={np.linalg.norm(k_solve * term3_bot):.4e}")

            result = np.concatenate([res_top, res_bot]) if (n_nongen > 0 or n_gen > 0) else np.array([],dtype=np.float64)
            if debug: print(f"  S22 matvec final result: Total Norm={np.linalg.norm(result):.4e}")

            return result
        S22_matvec = S22_ssgblup_matvec_with_debug


        # Define transformation operator T = diag(I_fixed, I_nongen, G)
        def T_ssgblup_matvec(v):
             res = np.zeros_like(v)
             res[:n_fixed + n_nongen] = v[:n_fixed + n_nongen] # I * [beta; u1]
             if n_gen > 0:
                  res[n_fixed + n_nongen:] = G_solve @ v[n_fixed + n_nongen:] # G * u2_gamma
             return res
        transform_T = LinearOperator((n_fixed+n_random, n_fixed+n_random), matvec=T_ssgblup_matvec, dtype=np.float64)

        # Define T' operator (since G is symmetric, T'=T)
        def T_prime_ssgblup_matvec(v):
             res = np.zeros_like(v)
             res[:n_fixed + n_nongen] = v[:n_fixed + n_nongen] # I * [b_beta; b_u1]
             if n_gen > 0:
                  res[n_fixed + n_nongen:] = G_solve @ v[n_fixed + n_nongen:] # G * b_u2
             return res
        rhs_transform_T_prime = LinearOperator((n_fixed+n_random, n_fixed+n_random), matvec=T_prime_ssgblup_matvec, dtype=np.float64)


    # --- Construct complete transformed system operator S ---
    S12_op = LinearOperator((n_fixed, n_random), matvec=S12_matvec, dtype=np.float64)
    S21_op = LinearOperator((n_random, n_fixed), matvec=S21_matvec, dtype=np.float64)
    S22_op = LinearOperator((n_random, n_random), matvec=S22_matvec, dtype=np.float64)

    def sp_Linearsys_matvec(v_gamma): # v_gamma is the transformed solution vector
        v_beta = v_gamma[:n_fixed]
        v_u_gamma = v_gamma[n_fixed:] # Note this is the gamma_u part
        res_top = S11_op.matvec(v_beta) + S12_op.matvec(v_u_gamma)
        res_bot = S21_op.matvec(v_beta) + S22_op.matvec(v_u_gamma)
        return np.concatenate([res_top, res_bot])

    # Linear operator for the complete transformed system S
    S_global_op = LinearOperator(
        (n_fixed + n_random, n_fixed + n_random),
        matvec=sp_Linearsys_matvec, dtype=np.float64
    )
    monitor_matrix_stats(S_global_op, f"Complete Linear System Operator S (Model: {model_type})")

    # --- Construct right-hand side vector b' = T' b ---
    print("\nConstructing right-hand side vector...")
    b_beta = X_s.T @ Y_s
    b_u = Z_s.T @ Y_s

    if model_type == 'ABLUP':
        # T' = I, b' = b
        rhs_final = np.concatenate([b_beta, b_u])
        print("RHS (ABLUP): [X'y, Z'y]")
    elif model_type == 'GBLUP':
        # T' = diag(I, G), b = [X'y, Zg'y]
        # T'b = [X'y, G Zg'y]
        rhs_final = np.concatenate([b_beta, G_solve @ b_u]) # b_u is Zg'y here
        print("RHS (GBLUP, Transformed): [X'y, G Zg'y]")
    elif model_type == 'ssGBLUP':
        # T' = diag(I, I, G), b = [X'y, Z1'y, Z2'y]
        # T'b = [X'y, Z1'y, G Z2'y]
        b_u1 = b_u[:n_nongen] if n_nongen > 0 else []
        b_u2 = b_u[n_nongen:] if n_gen > 0 else []
        rhs_transformed_u2 = G_solve @ b_u2 if n_gen > 0 else []
        rhs_final = np.concatenate([b_beta, b_u1, rhs_transformed_u2])
        print("RHS (ssGBLUP, Transformed): [X'y, Z1'y, G Z2'y]")

    monitor_matrix_stats(rhs_final, f"Final Right-Hand Side Vector RHS (Model: {model_type})")

# Inside build_and_solve_linear_system function

# --- Construct Preconditioner M^-1 (Approximation using Z'Z + k*diag(A^-1)) ---
    print("\nConstructing preconditioner (Approximation using Z'Z + k*diag(A^-1))...")
    M_prec = None
    try:
        D_inv_diag_approx = None # Diagonal for approximation
        if model_type in ['ABLUP', 'ssGBLUP']:
            if diag_A_inv_vector is None:
                 # If diag(A^-1) is not pre-calculated, can raise error or use D^-1 diagonal as alternative
                 print("Warning: diag(A^-1) vector not provided, will use diag(D^-1) for preconditioner approximation.")
                 if D_inv_full is None: raise ValueError("Missing D_inv_full to get diagonal.")
                 D_inv_diag_approx = D_inv_full.diagonal()
                 if len(D_inv_diag_approx) != n_random: raise ValueError("D_inv_full diagonal length does not match n_random.")
            else:
                 D_inv_diag_approx = diag_A_inv_vector # Use calculated diag(A^-1)
                 if len(D_inv_diag_approx) != n_random: raise ValueError("Provided diag_A_inv_vector length does not match n_random.")

        elif model_type == 'GBLUP':
             # GBLUP model does not use A^-1, can approximate with identity or zero vector
             print("  Warning: GBLUP model, preconditioner will use k=0 or k*I approximation (depending on implementation).")
             # Use k*I approximation (i.e., diag(A^-1) approximates to I)
             #D_inv_diag_approx = np.ones(n_random, dtype=target_dtype) # n_random = n_gen
             D_inv_diag_approx = np.ones(n_random, dtype=np.float64)


        Z_for_approx = Z_solve

        M_prec = ImprovedBlockPreconditioner(
            A_UL=S11_op,
            A_BR=S22_op,
            Z_for_S22=Z_for_approx,
            k_val=k_solve,
            # Pass diagonal for approximation (could be diag(A^-1) or diag(D^-1) or I)
            diag_A_inv_for_S22=D_inv_diag_approx,
            omega=preconditioner_omega,
            debug=debug_solver
        )
        print(f"Block diagonal preconditioner constructed successfully (using Z'Z + k*diag(A^-1) approximation, Omega={preconditioner_omega:.2f}).")

    except Exception as e:
        print(f"Failed to construct preconditioner: {e}. Preconditioner will not be used.")
        traceback.print_exc()
        M_prec = None

    # --- Load or Initialize Iteration State ---
    print("\nPreparing for iterative solving...")
    state_loaded = load_state(state_file_solve)
    initial_iter = 0
    # Initial solution gamma_0
    gamma_0 = np.zeros(S_global_op.shape[1], dtype=np.float64)

    if state_loaded:
        loaded_iter = state_loaded.get('iteration', 0)
        loaded_gamma = state_loaded.get('xk') # Saved state variable name is 'xk'
        if loaded_gamma is not None and len(loaded_gamma) == S_global_op.shape[1]:
            initial_iter = loaded_iter
            gamma_0 = np.asarray(loaded_gamma, dtype=np.float64)
            print(f"Loaded from state file, starting iteration: {initial_iter}.")
            # Can calculate residual of loaded solution here
            if debug_solver:
                initial_resid = np.linalg.norm(rhs_final - S_global_op @ gamma_0)
                initial_rel_resid = initial_resid / (np.linalg.norm(rhs_final) + 1e-15)
                print(f"  Initial relative residual of loaded solution: {initial_rel_resid:.4e}")
        else:
            print("Warning: Loaded state solution vector is invalid or dimensions do not match, starting from zero vector.")
            initial_iter = 0 # Force start from scratch
    else:
        print("No valid state file, starting from zero vector.")

    remaining_iter = max_iterations_solve - initial_iter
    if remaining_iter <= 0:
        print(f"Maximum iterations ({max_iterations_solve}) reached or exceeded. Returning current solution.")
        # Need to recover original solution x based on model
        x_final_solution = recover_original_solution(model_type, gamma_0, n_fixed, n_nongen, n_gen, transform_T, G_solve)
        return x_final_solution, gamma_0, 0 # Assume converged or max iterations reached last time saved

    # --- Create CG Callback Object ---
    cg_callback = CGCallback(
        system_matrix_cb=S_global_op,
        rhs_vector_cb=rhs_final,
        state_file_cb=state_file_solve,
        start_iter_cb=initial_iter, # Start counting from loaded iteration count
        total_max_iter_cb=max_iterations_solve,
        save_freq_cb=100, # Can adjust save frequency
        rtol=rtol_solve,
        debug=debug_solver
    )

    # --- Use SciPy CG to iteratively solve S * gamma = rhs_final ---
    print(f"\nStarting CG iterative solving (max {remaining_iter} iterations)...")
    gamma_solution = None
    solve_info = -99 # Default failure code

    try:
        print(f"\nCalling CG solver (max {remaining_iter} iterations)...")
        # *** Use rtol and atol, remove tol ***
        gamma_solution, info_code = spla.cg(
            S_global_op, rhs_final,
            x0=gamma_0,           # Initial solution
            rtol=rtol_solve,      # Relative tolerance
            atol=atol_solve,      # Absolute tolerance
            maxiter=remaining_iter, # Remaining iterations
            M=M_prec,             # Preconditioner
            callback=cg_callback  # Callback function
        )
        solve_info = info_code # Save exit code
        if solve_info == 0:
            print("\nCG solver converged successfully.")
        elif solve_info > 0:
            print(f"\nWarning: CG solver reached maximum iterations ({remaining_iter}) but did not reach desired precision. Info={solve_info}")
        else: # solve_info < 0
             print(f"\nError: CG solver failed due to input error or computational issues. Info={solve_info}")

    except Exception as e_cg:
        print(f"\nSevere error during CG solving: {e_cg}")
        traceback.print_exc()
        # Return loaded or previous best solution? Here return None to indicate solving failed
        x_final_solution = recover_original_solution(model_type, gamma_0, n_fixed, n_nongen, n_gen, transform_T, G_solve)
        return x_final_solution, gamma_0, -99 # Return last solution and error code

    # --- Solving finished, check results ---
    if gamma_solution is None: # If CG crashed internally and not assigned
        print("Error: CG failed to return solution vector gamma.")
        x_final_solution = recover_original_solution(model_type, gamma_0, n_fixed, n_nongen, n_gen, transform_T, G_solve)
        return x_final_solution, gamma_0, solve_info

    # Calculate final residual
    final_residual = np.linalg.norm(rhs_final - S_global_op @ gamma_solution)
    final_rel_residual = final_residual / (np.linalg.norm(rhs_final) + 1e-15)
    print(f"CG finished. Final absolute residual: {final_residual:.4e}, Final relative residual: {final_rel_residual:.4e}")

    # --- Save final state ---
    save_state(cg_callback.iteration, gamma_solution, final_rel_residual, state_file_solve)

    # --- Recover original solution vector x = [beta_hat, u_hat] ---
    print("\nRecovering original solution vector x = [beta_hat, u_hat]...")
    x_final_solution = recover_original_solution(model_type, gamma_solution, n_fixed, n_nongen, n_gen, transform_T, G_solve)

    # --- [Optional] Debugging and analysis of final results ---
    if debug_solver:
         # For example, can call analyze_residuals_by_generation
         # Needs to pass original Z matrix (Z_solve), and mapping info from individuals to generations/genotypes
         # (This information needs to be passed via solver_kwargs)
         ped_data_for_calc = solver_kwargs.get('ped_data_for_calc')
         genotyped_0_indices_in_full_ped = solver_kwargs.get('genotyped_0_indices_in_full_ped')
         generation_observed = solver_kwargs.get('generation_observed') # Observed generation indices
         unique_generations = solver_kwargs.get('unique_generations') # List of unique generation indices
         # Needs a map from observation to animal
         obs_to_animal_map = solver_kwargs.get('obs_to_animal_map') # dict: obs_idx -> animal_idx_in_ped

         if all(v is not None for v in [generation_observed, unique_generations, obs_to_animal_map]):
              print("\n[Debug] Analyzing final residuals by generation (based on original MME)...")
              # Needs to construct original MME operator (or approximation) to calculate residual r = b - Ax
              # This is more complex, omitted for now, or just analyze residuals of gamma solution
              # analyze_residuals_by_generation(...)
              pass
         else:
              print("[Debug] Insufficient information for generation-wise residual analysis.")


         # Analyze final solution vector
         if x_final_solution is not None:
              print("[Debug] Analyzing final solution vector x ...")
              hat_beta = x_final_solution[:n_fixed]
              hat_u = x_final_solution[n_fixed:]
              fe_names = solver_kwargs.get('fe_names', [f'FE{i+1}' for i in range(n_fixed)])
              # analyze_final_results(hat_beta, hat_u, ped_data_for_calc, genotyped_0_indices_in_full_ped, fe_names, solver_kwargs.get('has_pheno'))
              print(f"  Fixed effects estimates (first 5): {hat_beta[:5]}")
              print(f"  Random effects estimates (first 5): {hat_u[:5]}")
              print(f"  Random effects statistics: Mean={np.mean(hat_u):.4f}, Std Dev={np.std(hat_u):.4f}, Min={np.min(hat_u):.4f}, Max={np.max(hat_u):.4f}")


    total_solve_time = time.time() - solve_overall_start_time
    print(f"\nTotal linear system construction and solving time: {total_solve_time:.2f} seconds")

    return x_final_solution, gamma_solution, solve_info


def recover_original_solution(model_type, gamma_solution, n_fixed, n_nongen, n_gen, transform_T_op, G_matrix):
    """
    Recovers the original solution x = [beta_hat, u_hat] based on model type and transformed solution gamma.

    Parameters:
    model_type (str): 'ABLUP', 'GBLUP', 'ssGBLUP'.
    gamma_solution (np.ndarray): Transformed solution vector from CG solver.
    n_fixed (int): Number of fixed effects.
    n_nongen (int): Number of non-genotyped individuals (ssGBLUP only).
    n_gen (int): Number of genotyped individuals (GBLUP and ssGBLUP).
    transform_T_op (LinearOperator or None): Transformation T used (if None, then T=I).
    G_matrix (np.ndarray or np.memmap): G matrix (needed for GBLUP and ssGBLUP).

    Returns:
    np.ndarray or None: Recovered original solution vector x, None if recovery failed.
    """
    if gamma_solution is None:
        return None

    print(f"  Recovering original solution x from gamma (Model: {model_type})...")
    x_solution = None

    try:
        if model_type == 'ABLUP':
            # In ABLUP, T = I, gamma = [beta, u]
            x_solution = gamma_solution.copy() # Directly the original solution
            print("    ABLUP: x = gamma")
        elif model_type == 'GBLUP':
            # In GBLUP, T = diag(I, G), gamma = [beta, G^-1 ug]
            # x = T gamma = [beta, G * (G^-1 ug)] = [beta, ug]
            if G_matrix is None: raise ValueError("GBLUP solution recovery requires G matrix.")
            beta_hat = gamma_solution[:n_fixed]
            gamma_u = gamma_solution[n_fixed:]
            u_hat = G_matrix @ gamma_u # u_hat = G * gamma_u
            x_solution = np.concatenate([beta_hat, u_hat])
            print("    GBLUP: beta = gamma_beta; u = G @ gamma_u")
        elif model_type == 'ssGBLUP':
            # In ssGBLUP, T = diag(I, I_nn, G), gamma = [beta, u1, G^-1 u2]
            # x = T gamma = [beta, u1, G * (G^-1 u2)] = [beta, u1, u2] = [beta, u]
            if G_matrix is None: raise ValueError("ssGBLUP solution recovery requires G matrix.")
            beta_hat = gamma_solution[:n_fixed]
            u1_hat = gamma_solution[n_fixed : n_fixed + n_nongen] if n_nongen > 0 else []
            gamma_u2 = gamma_solution[n_fixed + n_nongen:] if n_gen > 0 else []
            u2_hat = G_matrix @ gamma_u2 if n_gen > 0 else []
            x_solution = np.concatenate([beta_hat, u1_hat, u2_hat])
            print("    ssGBLUP: beta = gamma_beta; u1 = gamma_u1; u2 = G @ gamma_u2")

        if x_solution is not None:
            monitor_matrix_stats(x_solution, "Recovered Original Solution Vector x")
        else:
            print("Error: Could not recover original solution vector.")

    except Exception as e:
        print(f"Error recovering original solution vector: {e}")
        traceback.print_exc()
        return None # Indicate recovery failure

    return x_solution

# --- Result Analysis and Log Functions ---
# (analyze_residuals_by_generation, analyze_final_results, generate_diagnostic_log)
# These functions were defined in previous code blocks, assuming they are available here.
# Note: Their comments also need to be translated to English.

def analyze_residuals_by_generation(gamma_solution, system_mat, rhs_vec,
                                   Z_mat, # Original Z matrix (n_obs x n_random)
                                   generation_data_all, # Generation information for all individuals (length n_random)
                                   obs_to_animal_map, # Dictionary: observation index -> animal 0-based index in full list
                                   debug=True):
    """
    [Debug Function] Analyzes residuals of the final solution gamma (r = b - S*gamma) by generation.
    Requires individual generation information and mapping from observations to individuals.

    Parameters:
    gamma_solution (np.ndarray): Final solution gamma obtained from CG solver.
    system_mat (LinearOperator): Transformed system matrix S.
    rhs_vec (np.ndarray): Transformed right-hand side vector b' = T'b.
    Z_mat (scipy.sparse.csr_matrix): Original random effects design matrix Z (n_obs x n_random).
    generation_data_all (np.ndarray): Array containing generation information for all n_random individuals.
    obs_to_animal_map (dict): Dictionary mapping observation row number to 0-based index of animal in full pedigree.
    debug (bool): Whether to print detailed information.
    """
    print("\n--- [Debug] Analyzing Residuals by Generation (based on gamma solution) ---")
    if gamma_solution is None:
        print("  Cannot analyze residuals: gamma solution vector is None.")
        return
    if system_mat is None or rhs_vec is None or Z_mat is None or generation_data_all is None or obs_to_animal_map is None:
        print("  Cannot analyze residuals: Missing required matrices, vectors, or mapping information.")
        return

    # --- Calculate global residual (based on gamma) ---
    try:
        residual_vector = rhs_vec - system_mat @ gamma_solution
        global_residual_norm = np.linalg.norm(residual_vector)
        global_relative_residual = global_residual_norm / (np.linalg.norm(rhs_vec) + 1e-15)
        print(f"  Global residual norm ||b' - S*gamma||: {global_residual_norm:.4e}")
        print(f"  Global relative residual: {global_relative_residual:.4e}")
    except Exception as e:
        print(f"  Error calculating global residual: {e}")
        return

    # --- Analyze by Generation ---
    # Need to map elements of the residual vector back to corresponding animals and their generations
    # The residual vector r = b' - S*gamma has dimension n_fixed + n_random
    # We are more concerned with the observation-related part, but this doesn't directly correspond to r.
    # Alternatively, we can analyze the original residual e = y - X*beta - Z*u
    # This requires first recovering beta_hat and u_hat
    # ... (This part of logic is more complex, so generation-wise analysis of gamma residuals is omitted for now) ...
    print("  Note: Implementation for generation-wise analysis of gamma residuals is complex and omitted here.")
    print("        It is recommended to analyze original residuals e = y - X*beta - Z*u after recovering the original solution x.")


def analyze_final_results(hat_beta_vals, hat_u_ebvs,
                         ped_data_for_calc, genotyped_0_indices_in_full_ped,
                         fe_names, has_pheno):
    """
    [Debug/Analysis Function] Detailed analysis of final results (fixed effects and breeding values),
    with special focus on the last generation.

    Parameters:
    hat_beta_vals (np.ndarray): Estimated fixed effects values.
    hat_u_ebvs (np.ndarray): Estimated breeding values (EBV) for all individuals.
    ped_data_for_calc (np.ndarray): Pedigree data used for calculation (includes generation info in 4th column).
    genotyped_0_indices_in_full_ped (list or np.ndarray): 0-based indices of genotyped individuals in the full list.
    fe_names (list): List of fixed effect names.
    has_pheno (np.ndarray): Boolean array, marking which individuals have phenotypic observations.
    """
    print("\n===== [Analysis] Detailed Analysis of Final Results =====")

    # --- Fixed Effects Analysis ---
    if hat_beta_vals is not None and len(hat_beta_vals) > 0:
        print("\nFixed Effect Estimates:")
        if len(fe_names) == len(hat_beta_vals):
            for name, val in zip(fe_names, hat_beta_vals):
                print(f"  {name}: {val:.6f}")
        else: # If name list does not match, just print values
            print("  (Fixed effect name list does not match number of values, only printing values)")
            for i, val in enumerate(hat_beta_vals):
                print(f"  Effect {i+1}: {val:.6f}")
    else:
        print("\nNo fixed effect estimates.")

    # --- Breeding Value (EBV) Analysis ---
    if hat_u_ebvs is not None and len(hat_u_ebvs) > 0:
        print("\nBreeding Value (EBV) Overall Statistics:")
        print(f"  Total individuals: {len(hat_u_ebvs)}")
        try:
            mean_ebv = np.mean(hat_u_ebvs)
            std_ebv = np.std(hat_u_ebvs)
            min_ebv = np.min(hat_u_ebvs)
            max_ebv = np.max(hat_u_ebvs)
            zero_ratio = np.mean(np.abs(hat_u_ebvs) < 1e-10) # Check proportion close to zero

            print(f"  Mean: {mean_ebv:.6f}")
            print(f"  Standard Deviation: {std_ebv:.6f}")
            print(f"  Minimum: {min_ebv:.6f}")
            print(f"  Maximum: {max_ebv:.6f}")
            print(f"  Proportion close to zero: {zero_ratio:.2%}")

            if np.any(np.isnan(hat_u_ebvs)) or np.any(np.isinf(hat_u_ebvs)):
                 print("  Warning: Breeding values contain NaN or Inf values!")

        except Exception as e:
            print(f"  Error calculating breeding value statistics: {e}")

        # --- Special analysis for genotyped individuals ---
        if genotyped_0_indices_in_full_ped is not None and len(genotyped_0_indices_in_full_ped) > 0:
            try:
                # Ensure indices are valid
                valid_geno_indices = [idx for idx in genotyped_0_indices_in_full_ped if 0 <= idx < len(hat_u_ebvs)]
                if len(valid_geno_indices) != len(genotyped_0_indices_in_full_ped):
                     print(f"  Warning: Some genotyped individual indices ({len(genotyped_0_indices_in_full_ped) - len(valid_geno_indices)} individuals) are out of breeding value array range.")

                if valid_geno_indices:
                    geno_ebvs = hat_u_ebvs[valid_geno_indices]
                    print("\nGenotyped Individual Breeding Value Statistics:")
                    print(f"  Sample size: {len(geno_ebvs)}")
                    print(f"  Mean: {np.mean(geno_ebvs):.6f}")
                    print(f"  Standard Deviation: {np.std(geno_ebvs):.6f}")
                    print(f"  Minimum: {np.min(geno_ebvs):.6f}")
                    print(f"  Maximum: {np.max(geno_ebvs):.6f}")
                    print(f"  Proportion close to zero: {np.mean(np.abs(geno_ebvs) < 1e-10):.2%}")
                else:
                     print("\nNo valid genotyped individual indices for analysis.")
            except Exception as e:
                 print(f"  Error analyzing genotyped individual breeding values: {e}")
        else:
             print("\nNo genotyped individual information available for special analysis.")

        # --- Analysis by Generation ---
        if ped_data_for_calc is not None and ped_data_for_calc.shape[0] == len(hat_u_ebvs) and ped_data_for_calc.shape[1] >= 4:
            print("\nBreeding Value Analysis by Generation:")
            # Use generation information from pedigree data (assuming in 4th column, 0-based index 3)
            generations = ped_data_for_calc[:, 3]
            unique_gens = sorted([g for g in np.unique(generations) if g >= 0]) # Get valid generations and sort

            if not unique_gens:
                 print("  No valid generation information found for analysis.")
            else:
                 for gen in unique_gens:
                    gen_mask = (generations == gen)
                    if np.any(gen_mask):
                        gen_ebvs = hat_u_ebvs[gen_mask]
                        print(f"\n  Generation {gen}:")
                        print(f"    Sample size: {len(gen_ebvs)}")
                        print(f"    Mean: {np.mean(gen_ebvs):.6f}")
                        print(f"    Standard Deviation: {np.std(gen_ebvs):.6f}")
                        print(f"    Minimum: {np.min(gen_ebvs):.6f}")
                        print(f"    Maximum: {np.max(gen_ebvs):.6f}")
                        zero_ratio_gen = np.mean(np.abs(gen_ebvs) < 1e-10)
                        print(f"    Proportion close to zero: {zero_ratio_gen:.2%}")

                        # --- Special focus on the last generation ---
                        if gen == unique_gens[-1]: # If it's the last generation
                             print("\n    【Last Generation】Detailed Information:")
                             # Check if zero value proportion is too high
                             if zero_ratio_gen > 0.5: # Arbitrary threshold, e.g., 50%
                                  print(f"    Warning: {zero_ratio_gen:.1%} of individuals in the last generation have EBVs close to zero, possibly indicating insufficient information or connectivity issues.")

                             # Check if this generation has phenotypes
                             if has_pheno is not None and len(has_pheno) == len(hat_u_ebvs):
                                 pheno_in_gen = has_pheno[gen_mask]
                                 pheno_ratio = np.mean(pheno_in_gen) if len(pheno_in_gen) > 0 else 0
                                 print(f"    Proportion of individuals with phenotypic records: {pheno_ratio:.1%}")
                                 if pheno_ratio < 0.1: # If phenotype proportion is very low
                                      print("    Hint: Proportion of individuals with phenotypes in the last generation is low.")

                             # Check if this generation has genotypes
                             if genotyped_0_indices_in_full_ped is not None:
                                  gen_indices_in_ped = np.where(gen_mask)[0]
                                  geno_in_gen_count = sum(1 for idx in gen_indices_in_ped if idx in genotyped_0_indices_in_full_ped)
                                  geno_ratio = geno_in_gen_count / len(gen_ebvs) if len(gen_ebvs) > 0 else 0
                                  print(f"    Proportion of individuals with genotypic records: {geno_ratio:.1%}")
                                  if geno_ratio < 0.1:
                                       print("    Hint: Proportion of individuals with genotypes in the last generation is low.")

                             # Check connectivity to previous generations (how many individuals have at least one known parent)
                             connected_count = 0
                             gen_indices_in_ped = np.where(gen_mask)[0]
                             for idx in gen_indices_in_ped:
                                 if ped_data_for_calc[idx, 1] >= 0 or ped_data_for_calc[idx, 2] >= 0:
                                     connected_count += 1
                             connect_ratio = connected_count / len(gen_ebvs) if len(gen_ebvs) > 0 else 0
                             print(f"    Proportion of individuals with at least one known parent: {connect_ratio:.1%}")
                             if connect_ratio < 0.5: # If connectivity is low
                                  print("    Warning: Pedigree connectivity to previous generations in the last generation is low, which might affect EBV estimation accuracy.")

                    else:
                         print(f"\n  Generation {gen}: No individuals.")
        else:
             print("\nCould not analyze breeding values by generation (missing valid pedigree or generation information).")
    else:
         print("\nNo breeding value (EBV) results available for analysis.")


def generate_diagnostic_log(log_file_path, **log_vars):
    """
    Generates a diagnostic log file containing run parameters, data statistics, and result summary.

    Parameters:
    log_file_path (str): Full path to the log file.
    log_vars (dict): Dictionary containing variables to be logged.
                     Expected key-value examples: 'model_type', 'h2', 'k_val', 'max_iterations_solve',
                     'final_estimated_total_pigs', 'G_matrix', 'num_actual_observations',
                     'num_fixed_effects', 'Diag_coeffs_val', 'solve_info', 'gamma_solution',
                     'x_solution', 'ped_data_for_calc', 'fe_names', 'start_time', 'end_time'
    """
    print(f"\nGenerating diagnostic log file: {log_file_path}")
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"===== IFBLUP Diagnostic Log ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) =====\n\n")

            # --- 1. System and Environment Information ---
            f.write("### 1. System and Environment ###\n")
            try:
                f.write(f"- Python Version: {sys.version.split()[0]}\n")
                f.write(f"- NumPy Version: {np.__version__}\n")
                f.write(f"- SciPy Version: {scipy.__version__}\n")
                f.write(f"- Pandas Version: {pd.__version__}\n")
                # f.write(f"- Numba Version: {numba.__version__}\n") # Requires numba import
                # f.write(f"- psutil Version: {psutil.__version__}\n")
                cpu_log = psutil.cpu_count(logical=True)
                cpu_phy = psutil.cpu_count(logical=False)
                f.write(f"- CPU Cores: {cpu_log} (Logical), {cpu_phy} (Physical)\n")
                mem_tot = psutil.virtual_memory().total / (1024**3)
                f.write(f"- Total Memory: {mem_tot:.2f} GB\n")
                # Log number of threads used (if environment variable set)
                threads = os.environ.get("OMP_NUM_THREADS", "N/A")
                f.write(f"- OMP_NUM_THREADS: {threads}\n")
            except Exception as e_sys:
                f.write(f"- Error getting system information: {e_sys}\n")
            f.write("\n")

            # --- 2. Model and Parameters ---
            f.write("### 2. Model and Parameters ###\n")
            f.write(f"- Model Type: {log_vars.get('model_type', 'N/A')}\n")
            f.write(f"- Heritability (h2): {log_vars.get('h2', 'N/A'):.4f}\n")
            f.write(f"- Variance Ratio (k): {log_vars.get('k_solve', 'N/A'):.6f}\n")
            f.write(f"- CG Max Iterations: {log_vars.get('max_iterations_solve', 'N/A')}\n")
            f.write(f"- CG Absolute Tolerance (atol): {log_vars.get('atol_solve', 'N/A'):.1e}\n")
            f.write(f"- CG Relative Tolerance (rtol): {log_vars.get('rtol_solve', 'N/A'):.1e}\n")
            f.write(f"- Preconditioner Omega: {log_vars.get('preconditioner_omega', 'N/A')}\n")
            f.write("\n")

            # --- 3. Data Statistics ---
            f.write("### 3. Data Statistics ###\n")
            n_total = log_vars.get('final_estimated_total_pigs', 'N/A')
            n_geno = log_vars.get('n_gen', 'N/A') # Get n_gen from log_vars
            n_nongen = log_vars.get('n_nongen', 'N/A') # Get n_nongen from log_vars
            n_obs = log_vars.get('num_actual_observations', 'N/A')
            n_fe = log_vars.get('num_fixed_effects', 'N/A')
            f.write(f"- Total Individuals (n_random): {n_total}\n")
            f.write(f"- Genotyped Individuals (n_gen): {n_geno}\n")
            f.write(f"- Non-Genotyped Individuals (n_nongen): {n_nongen}\n")
            f.write(f"- Valid Observations (n_obs): {n_obs}\n")
            f.write(f"- Fixed Effects Count (n_fe): {n_fe}\n")
            fe_names_list = log_vars.get('fe_names', [])
            if fe_names_list:
                 f.write(f"- Fixed Effect Names: {', '.join(fe_names_list)}\n")
            f.write("\n")

            # --- 4. Relationship Matrix Information (Sampled) ---
            f.write("### 4. Relationship Matrix Information (Sampled Statistics) ###\n")
            # G Matrix
            G_mat = log_vars.get('G_solve') # Note key name might need to be consistent
            if G_mat is not None and hasattr(G_mat, 'shape') and G_mat.shape[0] > 0:
                 g_size_log = G_mat.shape[0]
                 diag_sample_g = min(1000, g_size_log)
                 indices_g = np.random.choice(g_size_log, diag_sample_g, replace=False)
                 try:
                     diag_vals_g = np.array([G_mat[i,i] for i in indices_g])
                     f.write(f"- G Matrix Diagonal (Sampled {diag_sample_g}): Mean={np.mean(diag_vals_g):.4f}, Min={np.min(diag_vals_g):.4f}, Max={np.max(diag_vals_g):.4f}\n")
                 except Exception as e_gdiag: f.write(f"- G Matrix diagonal sampling failed: {e_gdiag}\n")
            else: f.write("- G Matrix: Not provided or empty\n")

            # D^-1 component of A^-1
            D_inv_mat = log_vars.get('D_inv_full')
            if D_inv_mat is not None and hasattr(D_inv_mat, 'shape') and D_inv_mat.shape[0] > 0:
                 try:
                     diag_vals_d = D_inv_mat.diagonal()
                     f.write(f"- D^-1 (from A^-1) Diagonal: Mean={np.mean(diag_vals_d):.4f}, Min={np.min(diag_vals_d):.4f}, Max={np.max(diag_vals_d):.4f}\n")
                 except Exception as e_ddiag: f.write(f"- D^-1 diagonal retrieval failed: {e_ddiag}\n")
            else: f.write("- D^-1 (from A^-1): Not provided or empty\n")

            # Inbreeding coefficient F
            F_coeffs_log = log_vars.get('Diag_coeffs_val') # Note key name
            if F_coeffs_log is not None and len(F_coeffs_log) > 0:
                 f.write(f"- Inbreeding Coefficient F: Mean={np.mean(F_coeffs_log):.4f}, Min={np.min(F_coeffs_log):.4f}, Max={np.max(F_coeffs_log):.4f}\n")
            else: f.write("- Inbreeding Coefficient F: Not provided\n")
            f.write("\n")

            # --- 5. Solver Information ---
            f.write("### 5. Solver Information ###\n")
            solve_info_log = log_vars.get('solve_info', 'N/A')
            f.write(f"- CG Solver Status Code: {solve_info_log} (0=Success, >0=Not converged, <0=Failure)\n")
            # Can get final iteration count and residual from callback object
            # final_iter = log_vars.get('final_iteration', 'N/A')
            # final_rel_res = log_vars.get('final_relative_residual', 'N/A')
            # f.write(f"- CG Final Iteration: {final_iter}\n")
            # f.write(f"- CG Final Relative Residual: {final_rel_res:.4e}\n")

            gamma_sol = log_vars.get('gamma_solution')
            if gamma_sol is not None:
                 f.write(f"- gamma Solution Vector Statistics: Norm={np.linalg.norm(gamma_sol):.4e}, Mean={np.mean(gamma_sol):.4e}, Min={np.min(gamma_sol):.4e}, Max={np.max(gamma_sol):.4e}\n")
            else: f.write("- gamma Solution Vector: Not obtained\n")
            f.write("\n")

            # --- 6. Final Solution (x) Statistics ---
            f.write("### 6. Final Solution (x = [beta, u]) Statistics ###\n")
            x_sol = log_vars.get('x_solution')
            if x_sol is not None:
                 n_fe = log_vars.get('num_fixed_effects', 0)
                 x_beta = x_sol[:n_fe] if n_fe > 0 else []
                 x_u = x_sol[n_fe:]
                 f.write(f"- x Solution Vector Total Length: {len(x_sol)}\n")
                 f.write(f"- Fixed Effects (beta) Statistics: Norm={np.linalg.norm(x_beta):.4e}, Mean={np.mean(x_beta):.4e}\n")
                 f.write(f"- Random Effects (u) Statistics: Norm={np.linalg.norm(x_u):.4e}, Mean={np.mean(x_u):.4f}, Std={np.std(x_u):.4f}, Min={np.min(x_u):.4f}, Max={np.max(x_u):.4f}\n")
                 # Can add generation-wise u value statistics here (if ped_data_for_calc is provided)
            else: f.write("- x Solution Vector: Not obtained or recovery failed\n")
            f.write("\n")

            # --- 7. Run Time ---
            f.write("### 7. Run Time ###\n")
            start_t = log_vars.get('overall_start_time')
            end_t = log_vars.get('overall_end_time')
            if start_t and end_t:
                 f.write(f"- Total Run Time: {end_t - start_t:.2f} seconds\n")
            # Can log time for each major step
            # f.write(f"- G Matrix Construction Time: ...\n")
            # f.write(f"- Solving Time: ...\n")
            f.write("\n")

            f.write("===== Log End =====\n")
        print(f"Diagnostic log successfully written to: {log_file_path}")

    except Exception as e_log:
        print(f"Severe error generating diagnostic log: {e_log}")
        traceback.print_exc()


@njit
def _calculate_diag_A_inv_numba(n, L_indices, L_indptr, L_data, d_inv_diag):
    """
    [Numba JIT Core Function] Calculate diag(A^-1)
    Using L matrix in CSC format (L=T^-1 in A^-1 = L' D^-1 L)
    """
    diag_A_inv = np.zeros(n, dtype=d_inv_diag.dtype)
    for i in range(n): # Iterate through diagonal element index i of A^-1 (also column index of L)
        sum_sq = 0.0
        # Get non-zero element information for column i of matrix L (CSC format)
        start_ptr = L_indptr[i]
        end_ptr = L_indptr[i+1]
        # Iterate through non-zero elements L_ki in column i
        for k_ptr in range(start_ptr, end_ptr):
            k = L_indices[k_ptr] # Row index k
            L_ki = L_data[k_ptr] # Value of L[k, i]
            # Accumulate according to formula (L_ki)^2 * (D^-1)_kk
            # Note: L=T^-1, A^-1 = L' D^-1 L, so we use L_ki
            # And the diagonal of L is 1
            if k < n: # Ensure k is within the range of d_inv_diag
                 sum_sq += (L_ki**2) * d_inv_diag[k]
            # else: Theoretically k should not exceed the range

        diag_A_inv[i] = sum_sq
    return diag_A_inv

def calculate_diag_A_inv(L_inv_csr, D_inv_diag):
    """
    Calculate diagonal elements of A^-1.

    Parameters:
    L_inv_csr (scipy.sparse.csr_matrix): L = T^-1 matrix (CSR format).
    D_inv_diag (np.ndarray): Vector of diagonal elements of D^-1 matrix.

    Returns:
    np.ndarray: Vector of diagonal elements of A^-1.
    """
    print("  Starting calculation of diag(A^-1)...")
    n = L_inv_csr.shape[0]
    if len(D_inv_diag) != n:
        raise ValueError("Dimensions of L_inv and D_inv_diag do not match.")

    # Convert L to CSC format for efficient column access
    print("    Converting L to CSC format...")
    L_inv_csc = L_inv_csr.tocsc()
    print("    Conversion complete.")

    # Call Numba-accelerated core calculation function
    print("    Using Numba to calculate diagonal elements...")
    diag_A_inv_result = _calculate_diag_A_inv_numba(n, L_inv_csc.indices, L_inv_csc.indptr, L_inv_csc.data, D_inv_diag)
    print("  diag(A^-1) calculation complete.")
    print(f"    diag(A^-1) statistics: Mean={np.mean(diag_A_inv_result):.4f}, Min={np.min(diag_A_inv_result):.4f}, Max={np.max(diag_A_inv_result):.4f}")

    return diag_A_inv_result

# -----------------------------------------------------------------------------
# 7. Main Program Entry
# -----------------------------------------------------------------------------
def main():
    """
    Main execution function, coordinating data loading, matrix construction, model solving, and result saving.
    """
    print("\n===== Starting IFBLUP Main Program =====")
    overall_start_time = time.time() # Record overall start time
    global_monitor = EnhancedResourceMonitor() # Initialize resource monitor
    global_monitor.update("Program Initialization")

    # --- Model Selection ---
    # Set the model type to run here: 'ABLUP', 'GBLUP', or 'ssGBLUP'
    model_to_run = 'GBLUP' # <--- Modify model type here
    print(f"\nSelected model type: {model_to_run}")

    # --- Basic Parameter Settings ---
    h2 = 0.5  # Example heritability value
    # Calculate variance ratio k = sigma_e^2 / sigma_u^2 = (1-h2)/h2 based on heritability
    if 0 < h2 < 1.0:
        k_val = (1.0 - h2) / h2
    elif h2 == 1.0:
        k_val = 1e-12  # Avoid division by zero, use a very small value
        print("Warning: Heritability h2 is set to 1.0, variance ratio k will use a very small value.")
    elif h2 == 0.0:
        k_val = 1e12  # Heritability is 0, k approaches infinity, set to a very large value
        print("Warning: Heritability h2 is set to 0.0, variance ratio k will use a very large value.")
    else:
        raise ValueError("Heritability h2 must be in the range [0, 1].")

    print(f"Setting heritability h2 = {h2:.3f}, variance ratio k = {k_val:.4f}")

    # --- File Path Settings (absolute paths recommended) ---
    # Set the base directory for data and results
    # Please modify this path according to your actual environment
    absolute_base_dir = os.path.dirname(os.path.abspath(__file__))

    # Ensure the base directory exists, attempt to create it if not
    if not os.path.exists(absolute_base_dir):
        try:
            os.makedirs(absolute_base_dir, exist_ok=True) # exist_ok=True prevents error if directory already exists
            print(f"Notice: Base directory '{absolute_base_dir}' does not exist, attempted to create it.")
        except OSError as e:
            print(f"Error: Unable to create base directory '{absolute_base_dir}': {e}. Program may not be able to read/write files correctly.")
            # You can choose to exit the program here: return

    # Input file path settings
    # Genotype file base name (without path and extension)
    # The program will automatically look for <bed_file_base_name>.bed/.bim/.fam files
    bed_file_base_name = "5k" 
    
    # Pedigree and phenotype CSV filename (absolute path recommended)
    # If using relative path, it's relative to the current working directory
    pig_data_csv_file_name = os.path.join(absolute_base_dir, "5055quan.csv")

    bed_file_base = os.path.join(absolute_base_dir, bed_file_base_name)
    pig_data_csv_path = os.path.join(absolute_base_dir, pig_data_csv_file_name)

    # Output and intermediate file paths (using absolute paths and model type suffix)
    output_ebv_path = os.path.join(absolute_base_dir, f"5055_corrected_ebv_{model_to_run}.csv")
    output_fixed_path = os.path.join(absolute_base_dir, f"fixed_effects_corrected_{model_to_run}.csv")
    g_matrix_memmap_path = os.path.join(absolute_base_dir, 'G_matrix_memmap.dat')
    z_centered_memmap_path = os.path.join(absolute_base_dir, 'Z_centered_memmap.dat')
    iter_state_file_path = os.path.join(absolute_base_dir, f'iteration_state_{model_to_run}_corrected.pkl')
    inbreeding_path = os.path.join(absolute_base_dir, "diag_coeffs.npy") # Inbreeding coefficient file
    diagnostic_log_path = os.path.join(absolute_base_dir, f"ifblup_diagnostic_log_{model_to_run}.txt")

    # Removed all previous definitions of base_path, Google Colab checks, and related error handling logic

    print("\nFile path settings (using absolute paths):")
    print(f"  Data and results directory: {absolute_base_dir}")
    print(f"  Pedigree/phenotype CSV: {pig_data_csv_path}")
    if model_to_run in ['GBLUP', 'ssGBLUP']:
        print(f"  Genotype BED/BIM/FAM: {bed_file_base}.*")
        print(f"  G matrix Memmap: {g_matrix_memmap_path}")
        print(f"  Z matrix Memmap: {z_centered_memmap_path}")
    if model_to_run in ['ABLUP', 'ssGBLUP']:
        print(f"  Inbreeding coefficients Numpy: {inbreeding_path}")
    print(f"  Iteration state Pickle: {iter_state_file_path}")
    print(f"  Output EBV CSV: {output_ebv_path}")
    print(f"  Output FE CSV: {output_fixed_path}")
    print(f"  Diagnostic log: {diagnostic_log_path}")

    # --- Check if input files exist ---
    input_files_ok = True
    if not os.path.exists(pig_data_csv_path):
        print(f"Error: Pedigree/phenotype CSV file not found: {pig_data_csv_path}")
        input_files_ok = False
    if model_to_run in ['GBLUP', 'ssGBLUP']:
        if not os.path.exists(bed_file_base + ".bed"):
            print(f"Error: Genotype BED file not found: {bed_file_base}.bed")
            input_files_ok = False
        if not os.path.exists(bed_file_base + ".bim"):
            print(f"Error: Genotype BIM file not found: {bed_file_base}.bim")
            input_files_ok = False
        if not os.path.exists(bed_file_base + ".fam"):
            print(f"Error: Genotype FAM file not found: {bed_file_base}.fam")
            input_files_ok = False

    if not input_files_ok:
        print("Required input files are missing, program terminated.")
        return

    # --- Initialize variables (for finally block and logging) ---
    # (Same as previous version, repetitive code omitted)
    G_matrix = None
    Z_final_design = None
    X_final_design = None
    ped_data_full = None # Store complete pedigree data read from CSV (original IDs)
    ped_data_internal = None # Store processed pedigree data with internal indices
    id_to_index_map = None
    index_to_id_map = None
    genotyped_0_indices = None # Positions of genotyped individuals in internal index list
    F_coeffs_val = None
    L_inv_full = None
    D_inv_full = None
    L_inv_ag = None
    D_inv_ag = None
    x_solution = None
    gamma_solution = None
    solve_info = -99
    final_num_individuals = 0 # Final total number of individuals
    n_gen_final = 0
    n_nongen_final = 0
    num_fixed_effects = 0
    fe_names = []
    Y_observed_final = None
    obs_to_animal_map_final = None
    has_pheno_final = None
    bed_ids_int = np.array([], dtype=np.int64) # Initialize as empty array

    try:
        # --- Step 1: Read and process pedigree and phenotype data (required for all models) ---
        print(f"\n--- Reading and processing pedigree/phenotype data: {pig_data_csv_path} ---")
        # Assumed CSV column indices: 0=animal ID, 1=sire ID, 2=dam ID, 3=sex, 4=year group, 9=phenotype
        # Use LargeScaleCSVReader for reading (if CSV is very large)
        # Estimate number of rows (need to know approximate row count, or modify the reader)
        # Simplified: use Pandas for estimation first
        try:
            # Only read ID column to estimate row count and unique IDs
            df_ids_only = pd.read_csv(pig_data_csv_path, usecols=[0], dtype=str) # Read animal ID column as string
            num_csv_rows = len(df_ids_only)
            print(f"CSV file row count (estimated): {num_csv_rows}")
            # Assume first row is header
            estimated_csv_pigs = num_csv_rows # Adjust if no header
            num_csv_cols = 13 # Assumed column count based on original code, or determine by reading header
            print(f"Assumed individual count approximately {estimated_csv_pigs}, CSV columns {num_csv_cols}")
        except Exception as e_count:
            print(f"Unable to estimate CSV row count: {e_count}. Cannot continue.")
            return

        # Create reader instance (if file is not large, can also use Pandas to read all at once)
        # PIG_DATA_READER = LargeScaleCSVReader(pig_data_csv_path, (estimated_csv_pigs, num_csv_cols), dtype=np.float64)
        # For simplification and to use your previous code, we temporarily still use Pandas to read all data
        try:
            # Note: column indices here are based on your original get_column calls
            df_ped_pheno = pd.read_csv(pig_data_csv_path, header=None) # Assume no header, read by index
            # Rename columns for better understanding (based on indices 0, 1, 2, 3, 4, 9)
            df_ped_pheno.columns = [f'col_{i}' for i in range(df_ped_pheno.shape[1])]
            col_animal, col_sire, col_dam, col_sex, col_year, col_pheno = 'col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_9'

             # Clean and convert ID columns, transform non-numeric or empty values to '-1'
            df_ped_pheno[col_animal] = pd.to_numeric(df_ped_pheno[col_animal], errors='coerce').fillna(-1).astype(np.int64)
            df_ped_pheno[col_sire] = pd.to_numeric(df_ped_pheno[col_sire], errors='coerce').fillna(-1).astype(np.int64)
            df_ped_pheno[col_dam] = pd.to_numeric(df_ped_pheno[col_dam], errors='coerce').fillna(-1).astype(np.int64)
             # Convert other columns
            df_ped_pheno[col_sex] = pd.to_numeric(df_ped_pheno[col_sex], errors='coerce') # Keep NaN
            df_ped_pheno[col_year] = pd.to_numeric(df_ped_pheno[col_year], errors='coerce').fillna(-1).astype(np.int64)
            df_ped_pheno[col_pheno] = pd.to_numeric(df_ped_pheno[col_pheno], errors='coerce') # Keep NaN

            print(f"Successfully read {len(df_ped_pheno)} rows of data.")
        except Exception as e_read:
             print(f"Failed to read or process CSV file {pig_data_csv_path}: {e_read}")
             raise

        # --- Determine individual list and ID mapping ---
        all_ids_in_ped = pd.unique(df_ped_pheno[[col_animal, col_sire, col_dam]].values.ravel('K'))
        all_ids_in_ped = all_ids_in_ped[all_ids_in_ped > 0] # Only keep valid IDs

        if model_to_run in ['GBLUP', 'ssGBLUP']:
            try:
                fam_data = pd.read_csv(bed_file_base + ".fam", sep='\s+', header=None, usecols=[1], dtype=str)
                bed_ids_int = pd.to_numeric(fam_data[1], errors='coerce').fillna(-1).astype(np.int64)
                bed_ids_int = bed_ids_int[bed_ids_int > 0]
                print(f"Read {len(bed_ids_int)} valid genotyped individual IDs from FAM file.")
            except Exception as e_fam:
                 print(f"Failed to read FAM file {bed_file_base}.fam: {e_fam}. GBLUP/ssGBLUP may not be possible.")
                 raise

        final_unique_ids = np.unique(np.concatenate((all_ids_in_ped, bed_ids_int)))
        final_num_individuals = len(final_unique_ids)
        if final_num_individuals == 0: raise ValueError("Could not find any valid individual IDs from data sources.")

        id_to_index_map = {original_id: index for index, original_id in enumerate(final_unique_ids)}
        index_to_id_map = {index: original_id for original_id, index in id_to_index_map.items()}
        print(f"Total number of individuals finally determined: {final_num_individuals}")
        print(f"Mapping created for Original ID <-> internal index [0..{final_num_individuals-1}].")

        # --- Construct pedigree data with internal indices (N x 4) ---
        # (Code same as previous version, omitted)
        ped_data_internal = np.full((final_num_individuals, 4), -1, dtype=np.int64)
        ped_data_internal[:, 0] = np.arange(final_num_individuals)
        df_ped_pheno_indexed = df_ped_pheno[df_ped_pheno[col_animal].isin(id_to_index_map)]
        for _, row in tqdm(df_ped_pheno_indexed.iterrows(), total=len(df_ped_pheno_indexed), desc="Building internal pedigree", ncols=80):
             animal_id = row[col_animal]
             target_idx = id_to_index_map.get(animal_id)
             if target_idx is not None:
                 ped_data_internal[target_idx, 1] = id_to_index_map.get(row[col_sire], -1)
                 ped_data_internal[target_idx, 2] = id_to_index_map.get(row[col_dam], -1)
                 ped_data_internal[target_idx, 3] = int(row[col_year]) if pd.notna(row[col_year]) else -1

        # --- Process generation/year group to 0-based index ---
        # (Code same as previous version, omitted)
        valid_gen_mask = ped_data_internal[:, 3] >= 0
        if np.any(valid_gen_mask):
            unique_generations_raw = np.unique(ped_data_internal[valid_gen_mask, 3])
            gen_to_idx_map = {gen_raw: idx for idx, gen_raw in enumerate(unique_generations_raw)}
            print(f"  Mapped {len(unique_generations_raw)} valid generation/year groups to index 0-{len(unique_generations_raw)-1}")
            for i in range(final_num_individuals):
                raw_gen = ped_data_internal[i, 3]
                ped_data_internal[i, 3] = gen_to_idx_map.get(raw_gen, -1)
        else: print("  Warning: No valid generation/year group information found.")
        unique_generation_indices_final = np.unique(ped_data_internal[:, 3][ped_data_internal[:, 3]>=0])

        # --- Extract phenotype and fixed effect data (aligned to internal index) ---
        # (Code same as previous version, omitted)
        Y_all = np.full(final_num_individuals, np.nan, dtype=np.float64)
        Sex_all = np.full(final_num_individuals, np.nan, dtype=np.float64)
        for _, row in df_ped_pheno_indexed.iterrows():
            target_idx = id_to_index_map.get(row[col_animal])
            if target_idx is not None:
                Y_all[target_idx] = row[col_pheno]
                Sex_all[target_idx] = row[col_sex]

        # --- Identify genotyped individuals ---
        # (Code same as previous version, omitted)
        if model_to_run in ['GBLUP', 'ssGBLUP']:
            genotyped_original_ids = set(bed_ids_int)
            genotyped_0_indices = sorted([idx for id_val, idx in id_to_index_map.items() if id_val in genotyped_original_ids])
            n_gen_final = len(genotyped_0_indices)
            n_nongen_final = final_num_individuals - n_gen_final
            if n_gen_final == 0:
                 raise ValueError("No valid genotyped individuals found.")
            print(f"Identified {n_gen_final} genotyped individuals and {n_nongen_final} non-genotyped individuals.")
        else: # ABLUP
             n_gen_final = 0
             n_nongen_final = final_num_individuals


        global_monitor.update("Data reading and ID mapping complete")




        # --- Step 2: Conditional construction of relationship matrices ---
        # G matrix (GBLUP/ssGBLUP)
        if model_to_run in ['GBLUP', 'ssGBLUP']:
            print("\nBuilding G matrix (memory mode)...") # Modified print message
            # Note: Removed passing of G_memmap_path parameter, as building_G_matrix now handles it internally
            # The meaning of the overwrite parameter is also diminished here, as it's always rebuilt in memory
            G_matrix_raw = building_G_matrix(
                bed_file_base,
                # Z_memmap_path=z_centered_memmap_path, # This parameter is also no longer used in building_G_matrix
                # G_memmap_path=g_matrix_memmap_path, # No longer need to pass path for memmap
                snp_block_size=51000, # You can adjust these parameters as needed
                block_size_G=81000,   # For example, based on your 64w.py script
                overwrite=True, # Force rebuild in memory
                target_dtype=np.float64 # Ensure passing
            )
            if G_matrix_raw is None: raise RuntimeError("G matrix construction failed.")
            G_matrix = G_matrix_raw # G_matrix_raw is now np.ndarray

            # Diagonal adjustment (Bending) of G matrix
            if G_matrix is not None: # G_matrix is now np.ndarray
                bending_value = 0.01 # Value you set
                print(f"\nDiagonal adjustment (Bending) of G matrix, diagonal elements increased by: {bending_value}")
                # Since G_matrix is a NumPy array in memory, it can be modified directly
                print("  G matrix is a Numpy array in memory, modifying directly.")
                G_matrix.flat[::G_matrix.shape[0] + 1] += bending_value
                # np.ndarray does not need flush
                print(f"  Diagonal adjustment complete (memory operation).")
            
            debug_validate_G_matrix(G_matrix) # Now validating G_matrix in memory
            global_monitor.update("G matrix ready (memory mode, adjusted)")


        # A^-1 components (ABLUP/ssGBLUP)
        if model_to_run in ['ABLUP', 'ssGBLUP']:
            print("\nCalculating inbreeding coefficients F...")
            # (Code same as previous version, omitted)
            if os.path.exists(inbreeding_path) and not False: # Can add overwrite control
                 try:
                      F_coeffs_val = np.load(inbreeding_path)
                      if len(F_coeffs_val) != final_num_individuals: F_coeffs_val = None
                 except Exception: F_coeffs_val = None

            if F_coeffs_val is None:
                 ped_data_numba = np.ascontiguousarray(ped_data_internal)
                 F_coeffs_val = calculate_inbreeding_coefficients(ped_data_internal, final_num_individuals)
                 if F_coeffs_val is None or len(F_coeffs_val) != final_num_individuals:
                      raise RuntimeError("Inbreeding coefficient calculation failed or returned invalid result.")
                 np.save(inbreeding_path, F_coeffs_val)
                 print(f"Inbreeding coefficients calculated and saved to {inbreeding_path}")
            else: print(f"Successfully loaded inbreeding coefficients from {inbreeding_path}.")
            print(f"Inbreeding coefficient F statistics: Mean={np.mean(F_coeffs_val):.4f}")


            print("\nBuilding components of A^-1 (L and D^-1)...")
            L_inv_full = build_L_inv_from_pedigree(ped_data_internal)
            D_inv_full = build_D_inv_from_pedigree(ped_data_internal, F_coeffs_val)
            global_monitor.update("A^-1 components ready")

        # A_22_ped^-1 components (ssGBLUP only)
        if model_to_run == 'ssGBLUP':
            print("\nBuilding components of A_22_ped^-1 (L_ag and D_ag^-1)...")
            # (Code same as previous version, omitted)
            ped_data_genotyped_subset = ped_data_internal[genotyped_0_indices, :]
            map_full_idx_to_geno_subset_idx = {orig_idx: new_idx for new_idx, orig_idx in enumerate(genotyped_0_indices)}
            ped_data_for_A22 = np.full((n_gen_final, 4), -1, dtype=np.int64)
            ped_data_for_A22[:, 0] = np.arange(n_gen_final)
            for i in range(n_gen_final):
                 orig_sire = ped_data_genotyped_subset[i, 1]; orig_dam = ped_data_genotyped_subset[i, 2]
                 ped_data_for_A22[i, 1] = map_full_idx_to_geno_subset_idx.get(orig_sire, -1)
                 ped_data_for_A22[i, 2] = map_full_idx_to_geno_subset_idx.get(orig_dam, -1)
                 ped_data_for_A22[i, 3] = ped_data_genotyped_subset[i, 3]
            F_coeffs_genotyped = F_coeffs_val[genotyped_0_indices]
            L_inv_ag = build_L_inv_from_pedigree(ped_data_for_A22)
            D_inv_ag = build_D_inv_from_pedigree(ped_data_for_A22, F_coeffs_genotyped)
            # validate_A22_ped_inv(L_inv_ag, D_inv_ag) # (Function defined in part two)
            global_monitor.update("A_22_ped^-1 components ready")

        diag_A_inv_vector = None
        if model_to_run in ['ABLUP', 'ssGBLUP']:
            print("\nCalculating diagonal elements of A^-1...")
            if L_inv_full is None or D_inv_full is None:
                raise ValueError("Cannot calculate diag(A^-1), L_inv_full or D_inv_full is missing.")
            try:
                # Get diagonal vector from D_inv_full (CSR format)
                d_inv_diag_full = D_inv_full.diagonal()
                # Call new function to calculate
                diag_A_inv_vector = calculate_diag_A_inv(L_inv_full, d_inv_diag_full)
                # (Optional) Save diag(A^-1) result
                # np.save(os.path.join(base_path, "diag_A_inv.npy"), diag_A_inv_vector)
            except Exception as e_diagAinv:
                print(f"Failed to calculate diag(A^-1): {e_diagAinv}")
                raise # Or choose to continue without preconditioner

        # --- Step 3: Prepare model inputs (X, Y, Z matrices - filtered) ---
        print("\n--- Preparing model inputs (filtering missing phenotypes/fixed effects)... ---")
        # (Filtering logic same as previous version, omitted)
        has_pheno_mask = ~np.isnan(Y_all)
        observed_indices_internal = np.where(has_pheno_mask)[0]
        num_obs_initial = len(observed_indices_internal)
        if num_obs_initial == 0: raise ValueError("No valid phenotypic observations.")
        Y_observed_initial = Y_all[observed_indices_internal]
        Sex_observed_initial = Sex_all[observed_indices_internal]
        Generation_observed_initial = ped_data_internal[observed_indices_internal, 3]
        missing_fe_mask = np.isnan(Sex_observed_initial) | (Generation_observed_initial < 0)
        num_missing_fe = np.sum(missing_fe_mask)
        if num_missing_fe > 0:
            valid_obs_mask = ~missing_fe_mask
            Y_observed_final = Y_observed_initial[valid_obs_mask]
            final_observed_indices_internal = observed_indices_internal[valid_obs_mask]
            Sex_observed_final = Sex_observed_initial[valid_obs_mask]
            Generation_indices_final = Generation_observed_initial[valid_obs_mask]
            print(f"  Removed {num_missing_fe} observations with missing fixed effects.")
        else:
            Y_observed_final = Y_observed_initial; final_observed_indices_internal = observed_indices_internal
            Sex_observed_final = Sex_observed_initial; Generation_indices_final = Generation_observed_initial
        num_actual_observations = len(Y_observed_final)
        if num_actual_observations == 0: raise ValueError("No valid observations after filtering.")
        print(f"Final number of valid observations: {num_actual_observations}")
        has_pheno_final = np.zeros(final_num_individuals, dtype=bool)
        has_pheno_final[final_observed_indices_internal] = True

        # Construct Z matrix (adjusted by model)
        print("Constructing final Z matrix...")
        rows_z = np.arange(num_actual_observations)
        cols_z_full = final_observed_indices_internal # Corresponds to indices in the complete individual list
        data_z = np.ones(num_actual_observations, dtype=np.float64)
        if model_to_run == 'GBLUP':
             map_full_to_geno = {full_idx: geno_idx for geno_idx, full_idx in enumerate(genotyped_0_indices)}
             cols_z_gblup = np.array([map_full_to_geno.get(full_idx, -1) for full_idx in cols_z_full])
             valid_z_mask = cols_z_gblup >= 0
             if np.sum(~valid_z_mask) > 0:
                  print(f"Warning (GBLUP): {np.sum(~valid_z_mask)} observations correspond to individuals without genotypes and will be removed.")
                  # Need to synchronously update rows and cols of Y, X, Z
                  rows_z = rows_z[valid_z_mask]
                  cols_z_gblup = cols_z_gblup[valid_z_mask]
                  Y_observed_final = Y_observed_final[valid_z_mask]
                  Sex_observed_final = Sex_observed_final[valid_z_mask]
                  Generation_indices_final = Generation_indices_final[valid_z_mask]
                  num_actual_observations = len(Y_observed_final)
                  if num_actual_observations == 0: raise ValueError("No valid observations after GBLUP filtering.")
             Z_final_design = csr_matrix((data_z[valid_z_mask], (np.arange(num_actual_observations), cols_z_gblup)),
                                         shape=(num_actual_observations, n_gen_final), dtype=np.float64)
        else: # ABLUP, ssGBLUP
             Z_final_design = csr_matrix((data_z, (rows_z, cols_z_full)),
                                         shape=(num_actual_observations, final_num_individuals), dtype=np.float64)
        monitor_matrix_stats(Z_final_design, f"Final Z matrix (Model: {model_to_run})")

        # Construct X matrix (same as previous version, omitting repetition)
        print("Constructing final X matrix...")
        unique_gens_final = np.unique(Generation_indices_final)
        n_gens_final = len(unique_gens_final); n_gen_dummies = max(0, n_gens_final - 1)
        num_fixed_effects = 1 + 1 + n_gen_dummies
        fe_names = ['Intercept', 'Sex']
        X_final_design = np.zeros((num_actual_observations, num_fixed_effects), dtype=np.float64)
        X_final_design[:, 0] = 1.0 # Intercept
        sex_mean_final = np.nanmean(Sex_observed_final) if np.any(np.isnan(Sex_observed_final)) else 0
        X_final_design[:, 1] = np.nan_to_num(Sex_observed_final, nan=sex_mean_final) # Sex
        if n_gen_dummies > 0:
            base_gen = unique_gens_final[0]
            for i, gen in enumerate(unique_gens_final[1:]):
                 X_final_design[:, 2+i] = (Generation_indices_final == gen).astype(np.float64)
                 fe_names.append(f"Gen_{gen}_vs_{base_gen}")
        monitor_matrix_stats(X_final_design, "Final X matrix")
        # check_fe_design(X_final_design)

        global_monitor.update("Model matrices X, Y, Z ready")

        # --- Step 4: Call core solver ---
        print(f"\n--- Calling core solver (Model: {model_to_run}) ---")
        solver_debug_data = { # Data passed to callback and log
            "ped_data_for_calc": ped_data_internal,
            "genotyped_0_indices_in_full_ped": genotyped_0_indices if model_to_run == 'ssGBLUP' else None,
            "generation_observed": Generation_indices_final,
            "unique_generations": unique_generation_indices_final,
            "obs_to_animal_map": dict(zip(np.arange(num_actual_observations), final_observed_indices_internal)),
            "fe_names": fe_names,
            "has_pheno": has_pheno_final
        }

        x_solution, gamma_solution, solve_info = build_and_solve_linear_system(
            model_type=model_to_run,
            X_solve=X_final_design,
            Y_solve=Y_observed_final,
            Z_solve=Z_final_design,
            k_solve=k_val,
            L_inv_full=L_inv_full,
            D_inv_full=D_inv_full,
            G_solve=G_matrix,
            L_inv_ag=L_inv_ag,
            D_inv_ag=D_inv_ag,
            n_nongen=n_nongen_final,
            n_gen=n_gen_final,
            state_file_solve=iter_state_file_path,
            max_iterations_solve=15000,
            atol_solve=1e-10,
            rtol_solve=1e-8,
            preconditioner_omega=0.7,
            debug_solver=True,
            # Pass diag_A_inv_vector to the solver, which then passes it to the preconditioner
            diag_A_inv_vector=diag_A_inv_vector, # <-- New pass-through
            **solver_debug_data
        )

        global_monitor.update("Linear system solution complete")

        # --- Step 5: Process and save results ---
        # (Code same as previous version, omitting repetition)
        if x_solution is not None and solve_info >= 0:
             print("\n--- Saving results ---")
             n_fe = X_final_design.shape[1]
             expected_len = n_fe + (n_gen_final if model_to_run == 'GBLUP' else final_num_individuals)
             if len(x_solution) == expected_len:
                 hat_beta = x_solution[:n_fe]
                 hat_u = x_solution[n_fe:]
                 if n_fe > 0:
                      fe_df = pd.DataFrame([hat_beta], columns=fe_names)
                      fe_df.to_csv(output_fixed_path, index=False, float_format='%.8f')
                      print(f"Fixed effects saved to: {output_fixed_path}")
                 if len(hat_u) > 0:
                      if model_to_run == 'GBLUP':
                           output_ids = [index_to_id_map.get(genotyped_0_indices[i]) for i in range(len(hat_u))]
                      else:
                           output_ids = [index_to_id_map.get(i) for i in range(len(hat_u))]
                      ebv_df = pd.DataFrame({'Original_Animal_ID': output_ids,'Estimated_Breeding_Value': hat_u})
                      ebv_df.to_csv(output_ebv_path, index=False, float_format='%.8f')
                      print(f"\nEstimated Breeding Values (EBV) saved to: {output_ebv_path}")
                      # analyze_final_results(...) # Call analysis function
                 else: print("\nModel did not estimate random effects.")
             else: print(f"Error: Solution vector length ({len(x_solution)}) does not match expected ({expected_len}).")
        else: print("\nSolution failed or no valid solution returned, cannot save results.")

        print("\n===== Main program execution complete =====")

    except Exception as e_main:
        print(f"\n===== Serious error occurred during main program execution: =====")
        print(f"Error type: {type(e_main).__name__}")
        print(f"Error message: {e_main}")
        print("\nDetailed error traceback:")
        traceback.print_exc()
        print("="*50)

    finally:
        # --- Program end, generate final diagnostic log ---
        overall_end_time = time.time()
        # (Code same as previous version, omitting repetition)
        log_data = {
            'model_type': model_to_run, 'h2': h2, 'k_solve': k_val,
            'max_iterations_solve': 15000, 'atol_solve': 1e-10, 'rtol_solve': 1e-8,
            'preconditioner_omega': 0.7,
            'final_estimated_total_pigs': final_num_individuals,
            'n_gen': n_gen_final, 'n_nongen': n_nongen_final,
            'num_actual_observations': num_actual_observations, 'num_fixed_effects': num_fixed_effects,
            'G_solve': G_matrix, 'D_inv_full': D_inv_full, 'Diag_coeffs_val': F_coeffs_val,
            'solve_info': solve_info, 'gamma_solution': gamma_solution, 'x_solution': x_solution,
            'ped_data_for_calc': ped_data_internal, 'fe_names': fe_names,
            'overall_start_time': overall_start_time, 'overall_end_time': overall_end_time,
        }
        generate_diagnostic_log(diagnostic_log_path, **log_data)

# --- Clean up resources ---
# (Same code as in the previous version, omitting duplication)
# G_matrix is ​​now np.ndarray, Python's garbage collection will automatically handle it
# But explicit deletion and gc.collect() are harmless
        if 'G_matrix' in locals() and G_matrix is not None:
            del G_matrix
            gc.collect()
        print("G_matrix in memory has been cleaned up (if no longer referenced).")

        global_monitor.update("Program ended, resources cleaned up")
        print(f"\nTotal execution time: {overall_end_time - overall_start_time:.2f} seconds")
        print("===== Program ended =====")


# --- Program entry point ---
if __name__ == "__main__":
    # Set Numpy print options (optional)
    np.set_printoptions(precision=6, suppress=True)

    # Execute main function
    main()