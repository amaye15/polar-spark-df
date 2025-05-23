# Polar-Spark-DF

[![PyPI version](https://badge.fury.io/py/polar-spark-df.svg)](https://badge.fury.io/py/polar-spark-df)

High-performance converters between PySpark and Polars DataFrames with optimized memory usage.

## Features

- 🚀 **High Performance**: Optimized for speed and memory efficiency
- 🔄 **Bidirectional Conversion**: Convert from PySpark to Polars and vice versa
- 🧱 **Builder Pattern**: Fluent interface for easy configuration
- 📊 **Batched Processing**: Handle large datasets with configurable batch sizes
- 🏹 **Apache Arrow Support**: Use Arrow for even faster conversions when available
- 📈 **Benchmarking Tools**: Built-in utilities to measure performance
- 🧪 **Well Tested**: Comprehensive test suite ensures reliability
- 🔄 **Type Safety**: Robust handling of various data types

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

## Supported Data Types

The converter supports a wide range of data types:

| Category | PySpark Types | Polars Types |
|----------|---------------|--------------|
| Numeric | `IntegerType`, `LongType`, `FloatType`, `DoubleType`, `DecimalType` | `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64`, `Float32`, `Float64`, `Decimal` |
| String | `StringType` | `Utf8` |
| Boolean | `BooleanType` | `Boolean` |
| Temporal | `DateType`, `TimestampType` | `Date`, `Datetime`, `Time` |
| Complex | `ArrayType`, `StructType`, `MapType` | `List`, `Struct` |

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

## Benchmark Results

Benchmark run on 2025-05-23T12:46:13

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

## Performance Recommendations

Based on our benchmarks:

1. **Use Apache Arrow for PySpark to Polars**: Arrow-based conversion is consistently 2-3x faster for PySpark to Polars conversion
2. **Batch Size Matters**: Larger batch sizes significantly improve performance, especially for PySpark to Polars conversion
3. **Optimal Configuration**: 
   - For PySpark to Polars: `arrow=True` with `batch_size=10000` provides the best performance
   - For Polars to PySpark: `arrow=False` with `batch_size=10000` is slightly faster than using Arrow
4. **Memory Considerations**: Memory usage is generally low for both conversion directions with the recommended configurations

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

# Run data type tests
pytest tests/test_datatypes.py -v
```

## License

MIT License