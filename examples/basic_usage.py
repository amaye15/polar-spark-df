"""
Basic usage examples for the DataFrameConverter.
"""

import polars as pl
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

from polar_spark_df.converter import DataFrameConverter


def main():
    """Demonstrate basic usage of the DataFrameConverter."""
    # Create a SparkSession
    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("polar-spark-df-example")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    
    # Create a sample PySpark DataFrame
    schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), True),
        StructField("value", DoubleType(), True)
    ])
    
    data = [
        (1, "Alice", 10.5),
        (2, "Bob", 20.0),
        (3, "Charlie", 30.7),
        (4, "David", None),
        (5, "Eve", 50.2)
    ]
    
    spark_df = spark.createDataFrame(data, schema)
    
    print("Original PySpark DataFrame:")
    spark_df.show()
    
    # Convert PySpark DataFrame to Polars using the converter
    polars_df = (
        DataFrameConverter()
        .with_spark_df(spark_df)
        .with_batch_size(10000)
        .with_use_arrow(True)
        .to_polars()
    )
    
    print("\nConverted Polars DataFrame:")
    print(polars_df)
    
    # Create a sample Polars DataFrame
    polars_df2 = pl.DataFrame({
        "id": [10, 20, 30],
        "name": ["John", "Jane", "Jim"],
        "score": [95.5, 88.0, 76.5]
    })
    
    print("\nOriginal Polars DataFrame:")
    print(polars_df2)
    
    # Convert Polars DataFrame to PySpark using the converter
    spark_df2 = (
        DataFrameConverter()
        .with_polars_df(polars_df2)
        .with_spark_session(spark)
        .with_use_arrow(True)
        .to_spark()
    )
    
    print("\nConverted PySpark DataFrame:")
    spark_df2.show()
    
    # Clean up
    spark.stop()


if __name__ == "__main__":
    main()