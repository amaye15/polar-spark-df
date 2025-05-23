"""
Script to run benchmarks and generate performance metrics.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
import numpy as np
import polars as pl
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from polar_spark_df.converter import DataFrameConverter
from polar_spark_df.benchmark import (
    benchmark_spark_to_polars,
    benchmark_polars_to_spark,
    print_benchmark_results
)


def create_test_dataframes(spark, num_rows, num_cols):
    """
    Create test DataFrames for benchmarking.
    
    Args:
        spark: SparkSession
        num_rows: Number of rows
        num_cols: Number of columns
        
    Returns:
        tuple: (spark_df, polars_df)
    """
    print(f"Creating test DataFrames with {num_rows} rows and {num_cols} columns...")
    
    # Create schema
    schema_fields = [
        StructField("id", IntegerType(), False)
    ]
    
    # Add numeric columns
    for i in range(num_cols // 3):
        schema_fields.append(StructField(f"num_{i}", DoubleType(), True))
    
    # Add string columns
    for i in range(num_cols // 3):
        schema_fields.append(StructField(f"str_{i}", StringType(), True))
    
    # Add more numeric columns to reach the desired number
    for i in range(num_cols - len(schema_fields)):
        schema_fields.append(StructField(f"extra_{i}", DoubleType(), True))
    
    schema = StructType(schema_fields)
    
    # Generate data
    data = []
    for i in range(num_rows):
        row = [i]  # id
        
        # Add numeric values
        for _ in range(num_cols // 3):
            row.append(float(np.random.rand() * 100))
        
        # Add string values
        for _ in range(num_cols // 3):
            row.append(f"str_value_{np.random.randint(0, 1000)}")
        
        # Add more numeric values
        for _ in range(num_cols - len(row)):
            row.append(float(np.random.rand() * 100))
        
        data.append(tuple(row))
    
    # Create PySpark DataFrame
    spark_df = spark.createDataFrame(data, schema)
    
    # Create Polars DataFrame
    polars_data = {}
    for i, field in enumerate(schema_fields):
        col_data = [row[i] for row in data]
        polars_data[field.name] = col_data
    
    polars_df = pl.DataFrame(polars_data)
    
    return spark_df, polars_df


def run_benchmarks(args):
    """
    Run benchmarks with the specified parameters.
    
    Args:
        args: Command-line arguments
    """
    # Create SparkSession
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("polar-spark-df-benchmarks")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    
    try:
        # Create test DataFrames
        spark_df, polars_df = create_test_dataframes(
            spark, 
            args.num_rows, 
            args.num_cols
        )
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "num_rows": args.num_rows,
                "num_cols": args.num_cols,
                "batch_sizes": args.batch_sizes,
                "use_arrow": args.use_arrow
            },
            "spark_to_polars": {},
            "polars_to_spark": {}
        }
        
        # Run PySpark to Polars benchmarks
        print("\nBenchmarking PySpark to Polars conversion...")
        spark_to_polars_results = benchmark_spark_to_polars(
            spark_df,
            batch_sizes=args.batch_sizes,
            use_arrow=args.use_arrow
        )
        
        print("\nResults for PySpark to Polars conversion:")
        print_benchmark_results(spark_to_polars_results)
        
        # Store results
        results["spark_to_polars"] = {
            k: {
                "execution_time": v["execution_time"],
                "memory_used": v["memory_used"]
            }
            for k, v in spark_to_polars_results.items()
        }
        
        # Run Polars to PySpark benchmarks
        print("\nBenchmarking Polars to PySpark conversion...")
        polars_to_spark_results = benchmark_polars_to_spark(
            polars_df,
            spark,
            batch_sizes=args.batch_sizes,
            use_arrow=args.use_arrow
        )
        
        print("\nResults for Polars to PySpark conversion:")
        print_benchmark_results(polars_to_spark_results)
        
        # Store results
        results["polars_to_spark"] = {
            k: {
                "execution_time": v["execution_time"],
                "memory_used": v["memory_used"]
            }
            for k, v in polars_to_spark_results.items()
        }
        
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
        
        # Store best configurations
        results["best_configurations"] = {
            "spark_to_polars": {
                "config": best_spark_to_polars[0],
                "execution_time": best_spark_to_polars[1]["execution_time"],
                "memory_used": best_spark_to_polars[1]["memory_used"]
            },
            "polars_to_spark": {
                "config": best_polars_to_spark[0],
                "execution_time": best_polars_to_spark[1]["execution_time"],
                "memory_used": best_polars_to_spark[1]["memory_used"]
            }
        }
        
        # Save results to file
        os.makedirs("benchmark_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results/benchmark_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBenchmark results saved to {filename}")
        
        # Generate markdown table for README
        generate_markdown_table(results)
    
    finally:
        # Clean up
        spark.stop()


def generate_markdown_table(results):
    """
    Generate a markdown table from benchmark results.
    
    Args:
        results: Benchmark results
    """
    markdown = "## Benchmark Results\n\n"
    markdown += f"Benchmark run on {results['timestamp']}\n\n"
    markdown += f"Dataset: {results['parameters']['num_rows']} rows × {results['parameters']['num_cols']} columns\n\n"
    
    # PySpark to Polars table
    markdown += "### PySpark to Polars Conversion\n\n"
    markdown += "| Configuration | Time (s) | Memory (MB) |\n"
    markdown += "|--------------|----------|-------------|\n"
    
    for config, metrics in results["spark_to_polars"].items():
        time_str = f"{metrics['execution_time']:.4f}"
        memory_str = f"{metrics['memory_used']:.2f}"
        markdown += f"| {config} | {time_str} | {memory_str} |\n"
    
    # Polars to PySpark table
    markdown += "\n### Polars to PySpark Conversion\n\n"
    markdown += "| Configuration | Time (s) | Memory (MB) |\n"
    markdown += "|--------------|----------|-------------|\n"
    
    for config, metrics in results["polars_to_spark"].items():
        time_str = f"{metrics['execution_time']:.4f}"
        memory_str = f"{metrics['memory_used']:.2f}"
        markdown += f"| {config} | {time_str} | {memory_str} |\n"
    
    # Best configurations
    markdown += "\n### Best Configurations\n\n"
    
    best_s2p = results["best_configurations"]["spark_to_polars"]
    best_p2s = results["best_configurations"]["polars_to_spark"]
    
    markdown += f"- **PySpark to Polars**: {best_s2p['config']} - {best_s2p['execution_time']:.4f}s, {best_s2p['memory_used']:.2f} MB\n"
    markdown += f"- **Polars to PySpark**: {best_p2s['config']} - {best_p2s['execution_time']:.4f}s, {best_p2s['memory_used']:.2f} MB\n"
    
    # Save markdown to file
    with open("benchmark_results/benchmark_table.md", "w") as f:
        f.write(markdown)
    
    print(f"\nMarkdown table saved to benchmark_results/benchmark_table.md")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run benchmarks for polar-spark-df")
    
    parser.add_argument(
        "--num-rows",
        type=int,
        default=100000,
        help="Number of rows in test DataFrames (default: 100000)"
    )
    
    parser.add_argument(
        "--num-cols",
        type=int,
        default=10,
        help="Number of columns in test DataFrames (default: 10)"
    )
    
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[10000, 50000, 100000],
        help="Batch sizes to test (default: 10000 50000 100000)"
    )
    
    parser.add_argument(
        "--use-arrow",
        type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
        nargs="+",
        default=[True, False],
        help="Whether to use Arrow (default: True False)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmarks(args)