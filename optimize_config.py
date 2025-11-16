#!/usr/bin/env python3
"""
Configuration and optimization settings for audio processing.
"""

import os
import psutil
from multiprocessing import cpu_count


def get_optimal_settings():
    """
    Get optimal processing settings based on system capabilities.
    
    Returns:
        dict: Configuration settings
    """
    # Get system info
    num_cpus = cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Optimal settings based on system
    if num_cpus >= 8 and memory_gb >= 16:
        # High-end system
        settings = {
            'n_workers': min(num_cpus - 2, 12),  # Leave 2 cores for system
            'chunk_size': 20,
            'batch_size': 64,
            'use_gpu': True
        }
    elif num_cpus >= 4 and memory_gb >= 8:
        # Mid-range system
        settings = {
            'n_workers': min(num_cpus - 1, 6),
            'chunk_size': 15,
            'batch_size': 32,
            'use_gpu': True
        }
    else:
        # Low-end system
        settings = {
            'n_workers': max(1, num_cpus - 1),
            'chunk_size': 10,
            'batch_size': 16,
            'use_gpu': False
        }
    
    # Check for GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            settings['gpu_device'] = 'cuda'
            settings['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)} with {settings['gpu_memory']:.1f}GB")
        else:
            settings['use_gpu'] = False
            settings['gpu_device'] = 'cpu'
    except ImportError:
        settings['use_gpu'] = False
        settings['gpu_device'] = 'cpu'
    
    return settings


def set_optimization_flags():
    """
    Set environment variables for optimization.
    """
    # Set threading optimization for librosa/numpy
    os.environ['OMP_NUM_THREADS'] = str(min(cpu_count(), 4))
    os.environ['MKL_NUM_THREADS'] = str(min(cpu_count(), 4))
    os.environ['NUMBA_NUM_THREADS'] = str(min(cpu_count(), 4))
    
    # Set memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    print(f"🔧 Optimization flags set for {cpu_count()} CPU cores")


def configure_librosa_for_speed():
    """
    Configure librosa for faster processing.
    """
    try:
        import librosa
        # Set librosa to use faster FFT implementation
        import scipy.fft
        librosa.set_fftlib(scipy.fft)
        print("⚡ Librosa configured for fast FFT processing")
    except ImportError:
        print("⚠️  Could not configure librosa optimizations")


if __name__ == "__main__":
    print("=== System Optimization Settings ===")
    
    # Get and display settings
    settings = get_optimal_settings()
    
    print(f"💻 System: {cpu_count()} CPUs, {psutil.virtual_memory().total / (1024**3):.1f}GB RAM")
    print(f"🔧 Recommended workers: {settings['n_workers']}")
    print(f"📦 Chunk size: {settings['chunk_size']}")
    print(f"🎯 Batch size: {settings['batch_size']}")
    print(f"🚀 GPU acceleration: {settings['use_gpu']}")
    
    # Apply optimizations
    set_optimization_flags()
    configure_librosa_for_speed()
    
    print("\n✅ System optimized for audio processing!")
