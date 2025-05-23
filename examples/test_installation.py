"""
Test script to verify the polar-spark-df package installation.
"""
import os
import sys
import pandas as pd
import polars as pl
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

from polar_spark_df import DataFrameConverter

def test_basic_conversion():
    """Test basic conversion functionality."""
    print("Creating SparkSession...")
    spark = SparkSession.builder \
        .appName("TestPolarSparkDF") \
        .master("local[*]") \
        .getOrCreate()
    
    print("Creating test data...")
    # Create a simple PySpark DataFrame
    schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), True),
        StructField("value", DoubleType(), True)
    ])
    
    data = [
        (1, "Alice", 10.5),
        (2, "Bob", 20.7),
        (3, "Charlie", 30.9)
    ]
    
    spark_df = spark.createDataFrame(data, schema)
    print("PySpark DataFrame created:")
    spark_df.show()
    
    print("\nConverting PySpark DataFrame to Polars...")
    # Convert to Polars
    converter = DataFrameConverter()
    polars_df = (
        converter
        .with_spark_df(spark_df)
        .with_batch_size(10)
        .with_use_arrow(True)
        .to_polars()
    )
    
    print("Polars DataFrame:")
    print(polars_df)
    
    print("\nConverting Polars DataFrame back to PySpark...")
    # Convert back to PySpark
    spark_df2 = (
        converter
        .with_polars_df(polars_df)
        .with_spark_session(spark)
        .to_spark()
    )
    
    print("Converted back to PySpark DataFrame:")
    spark_df2.show()
    
    # Verify data is the same
    print("\nVerifying data integrity...")
    assert spark_df2.count() == spark_df.count()
    
    # Compare data values
    pd_df1 = spark_df.toPandas()
    pd_df2 = spark_df2.toPandas()
    
    # Sort both DataFrames by id to ensure consistent order
    pd_df1 = pd_df1.sort_values('id').reset_index(drop=True)
    pd_df2 = pd_df2.sort_values('id').reset_index(drop=True)
    
    # Ensure same column order
    pd_df2 = pd_df2[pd_df1.columns]
    
    # Now compare
    pd.testing.assert_frame_equal(pd_df1, pd_df2, check_dtype=False)
    
    print("✅ Test passed! Data integrity maintained through conversion.")
    
    # Stop SparkSession
    spark.stop()

if __name__ == "__main__":
    test_basic_conversion()