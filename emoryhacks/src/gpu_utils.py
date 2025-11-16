"""
GPU utilities for CUDA acceleration.
Optimized for GeForce GTX 1660 Super (6GB VRAM, Turing architecture).
"""
import os
from typing import Optional, Tuple
import numpy as np


def check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import cupy as cp
        return cp.cuda.is_available()
    except ImportError:
        return False


def get_gpu_info() -> Optional[dict]:
    """Get GPU information if CUDA is available."""
    if not check_cuda_available():
        return None
    
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=5 * 1024**3)  # Limit to 5GB for GTX 1660 Super (6GB total)
        
        device = cp.cuda.Device()
        props = device.attributes
        meminfo = cp.cuda.runtime.memGetInfo()
        
        return {
            "available": True,
            "device_name": cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode(),
            "compute_capability": f"{props['Major']}.{props['Minor']}",
            "total_memory_gb": meminfo[1] / (1024**3),
            "free_memory_gb": meminfo[0] / (1024**3),
            "device_id": device.id,
        }
    except Exception as e:
        print(f"Warning: Could not get GPU info: {e}")
        return None


def get_optimal_chunk_size(data_size: int, n_workers: int = 4, gpu_available: bool = False) -> int:
    """
    Calculate optimal chunk size for data processing.
    For GPU: larger chunks to maximize GPU utilization
    For CPU: smaller chunks to balance memory and parallelism
    """
    if gpu_available:
        # GPU prefers larger batches
        chunk_size = max(32, data_size // (n_workers * 2))
    else:
        # CPU prefers smaller chunks for better cache usage
        chunk_size = max(8, data_size // (n_workers * 4))
    return chunk_size


def get_optimal_workers() -> Tuple[int, int]:
    """
    Get optimal number of workers for CPU processing.
    Returns (n_cores, n_threads) where n_cores = physical cores, n_threads = logical processors.
    """
    try:
        import multiprocessing as mp
        n_cores = mp.cpu_count() // 2  # Physical cores (assuming hyperthreading)
        n_threads = mp.cpu_count()  # Logical processors
        # For this system: 4 cores, 8 threads
        return (4, 8)
    except Exception:
        return (4, 8)  # Default fallback


def to_gpu(array: np.ndarray, dtype=None):
    """Transfer numpy array to GPU."""
    if not check_cuda_available():
        return array
    try:
        import cupy as cp
        if dtype:
            array = array.astype(dtype)
        return cp.asarray(array)
    except Exception as e:
        print(f"Warning: Could not transfer to GPU: {e}")
        return array


def to_cpu(array) -> np.ndarray:
    """Transfer GPU array back to CPU."""
    if not check_cuda_available():
        return array
    try:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
    except Exception:
        return array


def clear_gpu_cache():
    """Clear GPU memory cache."""
    if check_cuda_available():
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        except Exception:
            pass



