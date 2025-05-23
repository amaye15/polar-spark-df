# Polar-Spark-DF

High-performance converters between PySpark and Polars DataFrames with optimized memory usage.

## Features

- 🚀 **High Performance**: Optimized for speed and memory efficiency
- 🔄 **Bidirectional Conversion**: Convert from PySpark to Polars and vice versa
- 🧱 **Builder Pattern**: Fluent interface for easy configuration
- 📊 **Batched Processing**: Handle large datasets with configurable batch sizes
- 🏹 **Apache Arrow Support**: Use Arrow for even faster conversions when available
- 📈 **Benchmarking Tools**: Built-in utilities to measure performance
- 🧪 **Well Tested**: Comprehensive test suite ensures reliability

## Installation

```bash
pip install polar-spark-df
```

## Quick Start

### Converting PySpark DataFrame to Polars

```python
from polar_spark_df import DataFrameConverter

# Convert PySpark DataFrame to Polars
polars_df = (
    DataFrameConverter()
    .with_spark_df(spark_df)
    .with_batch_size(10000)
    .with_use_arrow(True)
    .to_polars()
)
```

### Converting Polars DataFrame to PySpark

```python
from polar_spark_df import DataFrameConverter

# Convert Polars DataFrame to PySpark
spark_df = (
    DataFrameConverter()
    .with_polars_df(polars_df)
    .with_spark_session(spark)
    .with_schema(schema)  # Optional
    .with_use_arrow(True)
    .to_spark()
)
```

## Configuration Options

The `DataFrameConverter` class provides several configuration options:

| Method | Description |
|--------|-------------|
| `with_spark_df(spark_df)` | Set the PySpark DataFrame to convert |
| `with_polars_df(polars_df)` | Set the Polars DataFrame to convert |
| `with_spark_session(spark_session)` | Set the SparkSession to use |
| `with_schema(schema)` | Set the schema for PySpark DataFrame creation |
| `with_batch_size(batch_size)` | Set the batch size for processing (default: 100000) |
| `with_use_arrow(use_arrow)` | Whether to use Arrow for conversion (default: True) |
| `with_preserve_index(preserve_index)` | Whether to preserve the index (default: False) |
| `with_optimize_string_conversion(optimize)` | Whether to optimize string conversion (default: True) |
| `with_type_mapping(type_mapping)` | Set custom type mapping for conversion |

## Performance Optimization

### Batch Size

The batch size controls how many rows are processed at once. A larger batch size may improve performance but requires more memory.

```python
converter = (
    DataFrameConverter()
    .with_batch_size(50000)  # Process 50,000 rows at a time
    # ... other configuration
)
```

### Apache Arrow

Using Apache Arrow can significantly improve performance, especially for large datasets:

```python
converter = (
    DataFrameConverter()
    .with_use_arrow(True)  # Use Arrow for conversion (default)
    # ... other configuration
)
```

## Benchmarking

The package includes benchmarking utilities to help you find the optimal configuration for your data:

```python
from polar_spark_df.benchmark import benchmark_spark_to_polars, print_benchmark_results

# Run benchmarks with different configurations
results = benchmark_spark_to_polars(
    spark_df,
    batch_sizes=[10000, 50000, 100000],
    use_arrow=[True, False]
)

# Print results
print_benchmark_results(results)
```

## Examples

See the [examples](https://github.com/amaye15/polar-spark-df/tree/main/examples) directory for more detailed usage examples:

- [Basic Usage](https://github.com/amaye15/polar-spark-df/blob/main/examples/basic_usage.py)
- [Benchmarking](https://github.com/amaye15/polar-spark-df/blob/main/examples/benchmark_example.py)

## Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run benchmarks
pytest tests/test_benchmark.py -v
```

## License

MIT License