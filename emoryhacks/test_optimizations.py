"""
Test script to verify multicore and GPU optimizations.
"""
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gpu_utils import (
    check_cuda_available,
    get_gpu_info,
    get_optimal_workers,
    get_optimal_chunk_size,
)


def test_cpu_info():
    """Test CPU information."""
    print("=" * 60)
    print("CPU Information")
    print("=" * 60)
    cores, threads = get_optimal_workers()
    print(f"Physical cores: {cores}")
    print(f"Logical processors: {threads}")
    print(f"Hyperthreading: {'Enabled' if threads > cores else 'Disabled'}")
    print()


def test_gpu_info():
    """Test GPU information."""
    print("=" * 60)
    print("GPU Information")
    print("=" * 60)
    if check_cuda_available():
        info = get_gpu_info()
        if info:
            print(f"GPU Available: Yes")
            print(f"Device Name: {info['device_name']}")
            print(f"Compute Capability: {info['compute_capability']}")
            print(f"Total Memory: {info['total_memory_gb']:.2f} GB")
            print(f"Free Memory: {info['free_memory_gb']:.2f} GB")
            print(f"Device ID: {info['device_id']}")
        else:
            print("GPU Available: Yes (but info unavailable)")
    else:
        print("GPU Available: No")
        print("Note: Install CuPy for GPU support")
        print("  pip install cupy-cuda11x  # for CUDA 11.x")
        print("  pip install cupy-cuda12x  # for CUDA 12.x")
    print()


def test_chunk_sizing():
    """Test optimal chunk size calculation."""
    print("=" * 60)
    print("Optimal Chunk Sizing")
    print("=" * 60)
    data_sizes = [100, 1000, 10000]
    gpu_available = check_cuda_available()
    
    for size in data_sizes:
        cpu_chunk = get_optimal_chunk_size(size, n_workers=8, gpu_available=False)
        gpu_chunk = get_optimal_chunk_size(size, n_workers=8, gpu_available=True)
        print(f"Data size: {size}")
        print(f"  CPU chunk size: {cpu_chunk}")
        print(f"  GPU chunk size: {gpu_chunk}")
        print(f"  GPU available: {gpu_available}")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Optimization Test Suite")
    print("=" * 60 + "\n")
    
    test_cpu_info()
    test_gpu_info()
    test_chunk_sizing()
    
    print("=" * 60)
    print("Test Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("1. If GPU is not available, install CuPy:")
    print("   pip install cupy-cuda11x  # for CUDA 11.x")
    print("2. For GPU-accelerated ML, install cuML:")
    print("   conda install -c rapidsai cuml cudf cudatoolkit=11.8")
    print("3. See GPU_OPTIMIZATION.md for detailed instructions")


if __name__ == "__main__":
    main()

