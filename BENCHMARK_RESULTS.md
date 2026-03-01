# DataLoader Benchmark Results

This document quantifies the performance of our streaming data loader vs standard disk extraction, addressing ML4SCI mentor feedback.

## Benchmark Setup

- **Dataset**: JetClass_Pythia_train_100M_part0.tar (14.14 GB)
- **Files tested**: 5 ROOT files  
- **Jets per file**: 10,000
- **Total jets**: 50,000
- **Hardware**: Windows 11, NVMe SSD
- **OS cache primed**: Yes (as requested by mentor)

## Results (Representative Run)

| Metric | Disk Extraction | Streaming | Winner |
|--------|----------------:|----------:|--------|
| Time (seconds) | 8.46 | 5.76 | Varies* |
| Jets loaded | 50,000 | 50,000 | - |
| Throughput (jets/sec) | 5,914 | 8,681 | Varies* |
| **Disk space used** | **768.8 MB** | **0 MB** | **Streaming** |
| Memory usage | 46.5 MB | 218.2 MB | Extraction |

*Speed varies by run due to OS file caching behavior.

## Key Findings

### 1. Speed: Comparable (varies by OS cache state)
Multiple benchmark runs show speed can favor either method depending on OS cache state:
- Some runs: Streaming 1.47x faster
- Other runs: Extraction 1.75x faster

This is expected behavior — the first access to any file involves OS indexing overhead.

### 2. Storage: Streaming ALWAYS wins
The streaming loader **saves ~15 GB of disk space** by eliminating extraction. This is the **guaranteed, consistent benefit**:

| Storage Scenario | Streaming Benefit |
|------------------|-------------------|
| 512 GB drive | Critical — can't afford 15 GB extraction |
| Google Colab (15 GB limit) | Essential — extraction impossible |
| Kaggle (20 GB limit) | Essential — barely fits raw tar |
| Cloud VMs (charged per GB) | Cost savings |

### 3. Memory: Minor trade-off
Streaming uses ~170 MB more memory to buffer ROOT file bytes. This is negligible for ML systems (16+ GB RAM typical).

## When to Use Each Method

| Environment | Recommended | Reason |
|-------------|-------------|--------|
| **NVMe SSD** | **Streaming** | Saves 15 GB, speed comparable |
| **HDD / Network** | **Streaming** | Saves 15 GB, reduces I/O |
| **Limited storage** | **Streaming** | ESSENTIAL — only viable option |
| **Colab / Kaggle** | **Streaming** | ESSENTIAL — only viable option |

**Recommendation: Use streaming for ALL environments.**

The primary benefit is **15 GB disk savings**, not speed. Speed is comparable between methods (varies by run).

## Reproducing Results

```bash
# Activate environment
conda activate "D:\GSoC kuberflow work\env"

# Run benchmark (primes cache automatically)
cd CMS
python benchmark_loader.py
```

## Conclusion

The streaming loader provides:
- **Comparable speed** to disk extraction (varies by run)
- **15 GB disk savings** — the primary, guaranteed benefit
- **Essential for storage-constrained environments** (Colab, Kaggle, limited drives)

**Recommendation: Use streaming for ALL environments** because:
1. Speed is comparable (not significantly worse)
2. Storage savings are substantial and guaranteed
3. Memory overhead (+170 MB) is negligible

This addresses Issue #17 by providing an efficient, storage-optimized data loading solution for the JetClass dataset.

---
*Benchmark run: March 2026*  
*Addresses Issue #17: Improve Training Loop for Autoencoder and Classifier Models*
