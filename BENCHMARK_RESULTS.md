# DataLoader Benchmark Results

This document quantifies the performance of our streaming data loader vs standard disk extraction, addressing ML4SCI mentor feedback.

## Benchmark Setup

- **Dataset**: JetClass_Pythia_train_100M_part0.tar (14.14 GB)
- **Files tested**: 5 ROOT files
- **Jets per file**: 10,000
- **Total jets**: 50,000
- **Hardware**: Windows 11, NVMe SSD
- **OS cache primed**: Yes (as requested by mentor)

## Results

| Metric | Disk Extraction | Streaming | Winner |
|--------|----------------:|----------:|--------|
| Time (seconds) | 8.46 | **5.76** | Streaming |
| Jets loaded | 50,000 | 50,000 | - |
| Throughput (jets/sec) | 5,914 | **8,681** | Streaming |
| **Disk space used** | 768.8 MB | **0 MB** | Streaming |
| Memory delta | **46.5 MB** | 218.2 MB | Extraction |

## Key Finding

**Streaming is 1.47x FASTER than disk extraction** while saving ~15 GB of disk space.

## Analysis

### Speed Advantage
Streaming is faster because:
- No double I/O (extract-to-disk then read-from-disk)
- Data goes directly: tar → memory buffer → numpy arrays
- Avoids disk write latency

### Storage Efficiency
The streaming loader **saves ~15 GB of disk space** by eliminating extraction:

| Storage Scenario | Streaming Benefit |
|------------------|-------------------|
| 512 GB drive | Critical — can't afford 15 GB extraction |
| Google Colab (15 GB limit) | Essential — extraction impossible |
| Kaggle (20 GB limit) | Essential — barely fits raw tar |
| Cloud VMs (charged per GB) | Cost savings |

### Memory Trade-off
Streaming uses ~170 MB more memory because it buffers ROOT file bytes before parsing. This is acceptable for most systems (even Colab has 12+ GB RAM).

## When to Use Each Method

| Environment | Recommended Method | Reason |
|-------------|-------------------|--------|
| Any storage-constrained | **Streaming** | Saves 15 GB |
| Colab / Kaggle | **Streaming** | Avoids quota limits |
| HDD / Network drive | **Streaming** | Less I/O overhead |
| Repeated training runs | Disk extraction | One-time extraction cost |

## Reproducing Results

```bash
# Activate environment
conda activate "D:\GSoC kuberflow work\env"

# Run benchmark (primes cache automatically)
cd CMS
python benchmark_loader.py
```

## Conclusion

The streaming loader is:
- **1.47x faster** than disk extraction
- **Saves 15 GB** of disk space
- **Essential** for Colab/Kaggle and storage-constrained environments

This addresses Issue #17 by providing an efficient data loading solution for the JetClass dataset.

---
*Benchmark run: March 2026*  
*Addresses Issue #17: Improve Training Loop for Autoencoder and Classifier Models*
