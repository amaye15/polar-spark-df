"""
Benchmarking example for the DataFrameConverter.
"""

import polars as pl
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import numpy as np

from polar_spark_df.converter import DataFrameConverter
from polar_spark_df.benchmark import (
    benchmark_spark_to_polars,
    benchmark_polars_to_spark,
    print_benchmark_results
)


def main():
    """Run benchmarks for the DataFrameConverter."""
    # Create a SparkSession
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("polar-spark-df-benchmark")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    
    # Create a schema for the PySpark DataFrame
    schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), True),
        StructField("value1", DoubleType(), True),
        StructField("value2", DoubleType(), True),
        StructField("value3", DoubleType(), True)
    ])
    
    # Generate random data for PySpark DataFrame
    num_rows = 100000
    spark_data = []
    for i in range(num_rows):
        name = f"name_{i}"
        value1 = float(np.random.rand())
        value2 = float(np.random.rand() * 100)
        value3 = float(np.random.rand() * 1000)
        spark_data.append((i, name, value1, value2, value3))
    
    spark_df = spark.createDataFrame(spark_data, schema)
    
    # Generate random data for Polars DataFrame
    polars_df = pl.DataFrame({
        "id": range(num_rows),
        "name": [f"name_{i}" for i in range(num_rows)],
        "value1": np.random.rand(num_rows),
        "value2": np.random.rand(num_rows) * 100,
        "value3": np.random.rand(num_rows) * 1000
    })
    
    print(f"Created DataFrames with {num_rows} rows for benchmarking")
    
    # Benchmark PySpark to Polars conversion
    print("\nBenchmarking PySpark to Polars conversion...")
    spark_to_polars_results = benchmark_spark_to_polars(
        spark_df,
        batch_sizes=[10000, 50000, 100000],
        use_arrow=[True, False]
    )
    
    print("\nResults for PySpark to Polars conversion:")
    print_benchmark_results(spark_to_polars_results)
    
    # Benchmark Polars to PySpark conversion
    print("\nBenchmarking Polars to PySpark conversion...")
    polars_to_spark_results = benchmark_polars_to_spark(
        polars_df,
        spark,
        schema=schema,
        batch_sizes=[10000, 50000, 100000],
        use_arrow=[True, False]
    )
    
    print("\nResults for Polars to PySpark conversion:")
    print_benchmark_results(polars_to_spark_results)
    
    # Find the best configuration for each direction
    best_spark_to_polars = min(
        spark_to_polars_results.items(),
        key=lambda x: x[1]["execution_time"]
    )
    
    best_polars_to_spark = min(
        polars_to_spark_results.items(),
        key=lambda x: x[1]["execution_time"]
    )
    
    print("\nBest configuration for PySpark to Polars:")
    print(f"  {best_spark_to_polars[0]}")
    print(f"  Time: {best_spark_to_polars[1]['execution_time']:.4f} seconds")
    print(f"  Memory: {best_spark_to_polars[1]['memory_used']:.2f} MB")
    
    print("\nBest configuration for Polars to PySpark:")
    print(f"  {best_polars_to_spark[0]}")
    print(f"  Time: {best_polars_to_spark[1]['execution_time']:.4f} seconds")
    print(f"  Memory: {best_polars_to_spark[1]['memory_used']:.2f} MB")
    
    # Clean up
    spark.stop()


if __name__ == "__main__":
    main()