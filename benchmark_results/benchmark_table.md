## Benchmark Results

Benchmark run on 2025-05-23T12:46:02.224830

Dataset: 10000 rows × 5 columns

### PySpark to Polars Conversion

| Configuration | Time (s) | Memory (MB) |
|--------------|----------|-------------|
| arrow=True,batch_size=1000 | 2.3950 | 7.27 |
| arrow=True,batch_size=5000 | 0.2647 | 1.14 |
| arrow=True,batch_size=10000 | 0.2431 | 1.29 |
| arrow=False,batch_size=1000 | 6.0196 | 3.14 |
| arrow=False,batch_size=5000 | 0.9136 | 0.77 |
| arrow=False,batch_size=10000 | 0.5439 | 1.25 |

### Polars to PySpark Conversion

| Configuration | Time (s) | Memory (MB) |
|--------------|----------|-------------|
| arrow=True,batch_size=1000 | 0.0514 | 2.57 |
| arrow=True,batch_size=5000 | 0.0243 | 0.26 |
| arrow=True,batch_size=10000 | 0.0197 | 0.00 |
| arrow=False,batch_size=1000 | 0.1190 | 0.00 |
| arrow=False,batch_size=5000 | 0.0286 | 1.62 |
| arrow=False,batch_size=10000 | 0.0173 | 0.00 |

### Best Configurations

- **PySpark to Polars**: arrow=True,batch_size=10000 - 0.2431s, 1.29 MB
- **Polars to PySpark**: arrow=False,batch_size=10000 - 0.0173s, 0.00 MB
