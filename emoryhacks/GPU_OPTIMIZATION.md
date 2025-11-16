# GPU Optimization Guide

This project has been optimized for multicore CPU processing (4 physical cores, 8 logical processors) and GPU acceleration using CUDA (optimized for GeForce GTX 1660 Super).

## System Requirements

- **CPU**: 4 physical cores, 8 logical processors (hyperthreading enabled)
- **GPU**: NVIDIA GPU with CUDA support (tested on GTX 1660 Super with 6GB VRAM)
- **CUDA**: Version 11.x or 12.x (check your GPU's compute capability)

## Installation

### 1. Install Base Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install GPU Dependencies (Optional)

#### For CUDA 11.x (GTX 1660 Super compatible):

```bash
# Install CuPy for GPU array operations
pip install cupy-cuda11x

# Install RAPIDS cuML for GPU-accelerated ML (recommended via conda)
conda install -c rapidsai -c conda-forge -c nvidia cuml cudf cudatoolkit=11.8
```

#### For CUDA 12.x:

```bash
pip install cupy-cuda12x
# cuML installation similar, adjust cudatoolkit version
```

**Note**: GPU packages are large (several GB) and CUDA-version specific. The code will automatically fall back to CPU if GPU libraries are not available.

## Usage

### Automatic GPU Detection

The code automatically detects GPU availability and falls back to CPU if GPU libraries are not installed:

```python
from emoryhacks.src.preprocess import preprocess_directory
from emoryhacks.src.build_dataset import build_features_from_metadata
from emoryhacks.src.run_training import run_all

# GPU will be auto-detected and used if available
preprocess_directory(raw_dir, interim_dir)
build_features_from_metadata(metadata_csv, project_root, output_csv)
run_all(project_root, features_csv, splits_dir, rf_out, ensemble_out)
```

### Explicit GPU Control

You can explicitly control GPU usage:

```python
# Force GPU usage (will raise error if GPU not available)
preprocess_directory(raw_dir, interim_dir, n_workers=8)
build_features_from_metadata(metadata_csv, project_root, output_csv, use_gpu=True)
run_all(project_root, features_csv, splits_dir, rf_out, ensemble_out, use_gpu=True)

# Force CPU usage
build_features_from_metadata(metadata_csv, project_root, output_csv, use_gpu=False)
run_all(project_root, features_csv, splits_dir, rf_out, ensemble_out, use_gpu=False)
```

## Performance Optimizations

### 1. Multiprocessing (CPU)

- **Preprocessing**: Uses all 8 logical processors for parallel audio file processing
- **Feature Extraction**: Chunks data optimally for 4 cores/8 threads
- **Model Training**: Uses `n_jobs=-1` to utilize all CPU cores

### 2. GPU Acceleration

- **Feature Extraction**: GPU-accelerated array operations (delta computation, spectral features)
- **Model Training**: GPU-accelerated Random Forest via cuML (RAPIDS)
- **Memory Management**: Automatic GPU memory pool management (limited to 5GB for GTX 1660 Super)

### 3. Data Chunking

The system automatically determines optimal chunk sizes:
- **GPU**: Larger chunks to maximize GPU utilization
- **CPU**: Smaller chunks for better cache usage

## Checking GPU Status

```python
from emoryhacks.src.gpu_utils import check_cuda_available, get_gpu_info

if check_cuda_available():
    info = get_gpu_info()
    print(f"GPU: {info['device_name']}")
    print(f"Memory: {info['free_memory_gb']:.2f} GB free / {info['total_memory_gb']:.2f} GB total")
else:
    print("GPU not available, using CPU")
```

## Performance Expectations

### CPU-Only (8 logical processors):
- Preprocessing: ~2-5x speedup vs sequential
- Feature extraction: ~3-6x speedup vs sequential
- Model training: ~2-4x speedup vs single-core

### GPU-Accelerated (GTX 1660 Super):
- Feature extraction: Additional ~1.5-2x speedup over CPU multiprocessing
- Model training: Additional ~2-5x speedup for large datasets (1000+ samples)

**Note**: GPU acceleration is most beneficial for:
- Large datasets (>1000 audio files)
- Complex feature extraction pipelines
- Large model training (many estimators, deep trees)

## Troubleshooting

### GPU Not Detected

1. Check CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Verify CuPy installation:
   ```python
   import cupy as cp
   print(cp.cuda.is_available())
   ```

3. Check CUDA version compatibility:
   ```python
   import cupy as cp
   print(cp.cuda.runtime.runtimeGetVersion())
   ```

### Out of Memory Errors

- Reduce batch sizes in feature extraction
- Limit GPU memory pool (already set to 5GB for GTX 1660 Super)
- Process data in smaller chunks

### Performance Issues

- Ensure GPU is being used (check with `get_gpu_info()`)
- Verify multiprocessing is working (check CPU usage)
- Consider reducing `n_workers` if system becomes unresponsive

## Architecture Details

### CPU Optimization
- Uses `multiprocessing.Pool` with optimal worker count
- Chunks data to balance parallelism and memory usage
- Leverages hyperthreading (8 logical processors)

### GPU Optimization
- CuPy for GPU-accelerated NumPy operations
- RAPIDS cuML for GPU-accelerated sklearn-compatible models
- Automatic memory management and cleanup
- Fallback to CPU if GPU unavailable

## Files Modified/Created

- `src/gpu_utils.py`: GPU utilities and device management
- `src/features_gpu.py`: GPU-accelerated feature extraction
- `src/ml_train_gpu.py`: GPU-accelerated model training
- `src/preprocess.py`: Added multiprocessing support
- `src/build_dataset.py`: Added multiprocessing and GPU support
- `src/ensemble_train.py`: Added GPU support for Random Forest
- `src/run_training.py`: Added GPU option

All changes are backward compatible - existing code will work without GPU libraries installed.


