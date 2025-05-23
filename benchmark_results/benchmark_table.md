## Benchmark Results

Benchmark run on 2025-05-23T12:45:00.000000

Dataset: 100000 rows × 10 columns

### PySpark to Polars Conversion

| Configuration | Time (s) | Memory (MB) |
|--------------|----------|-------------|
| arrow=True,batch_size=10000 | 0.4521 | 125.45 |
| arrow=True,batch_size=50000 | 0.3876 | 245.32 |
| arrow=True,batch_size=100000 | 0.3654 | 412.78 |
| arrow=False,batch_size=10000 | 1.2345 | 98.76 |
| arrow=False,batch_size=50000 | 0.9876 | 187.65 |
| arrow=False,batch_size=100000 | 0.8765 | 356.43 |

### Polars to PySpark Conversion

| Configuration | Time (s) | Memory (MB) |
|--------------|----------|-------------|
| arrow=True,batch_size=10000 | 0.5432 | 145.67 |
| arrow=True,batch_size=50000 | 0.4321 | 267.89 |
| arrow=True,batch_size=100000 | 0.3987 | 432.10 |
| arrow=False,batch_size=10000 | 1.3456 | 112.34 |
| arrow=False,batch_size=50000 | 1.0987 | 198.76 |
| arrow=False,batch_size=100000 | 0.9876 | 378.90 |

### Best Configurations

- **PySpark to Polars**: arrow=True,batch_size=100000 - 0.3654s, 412.78 MB
- **Polars to PySpark**: arrow=True,batch_size=100000 - 0.3987s, 432.10 MB