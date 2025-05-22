import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col, when, regexp_replace, split,to_date, lit
from pyspark.sql.utils import AnalysisException
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_gold_table(snapshot_date_str, silver_clickstream_directory, silver_attributes_directory, silver_financial_directory, gold_table_directory, spark):
    # Parse snapshot_date
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Load Silver Clickstream Table
    clickstream_partition_name = f"silver_feat_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    clickstream_filepath = os.path.join(silver_clickstream_directory, clickstream_partition_name)
    if not os.path.exists(clickstream_filepath):
        print(f"File not found: {clickstream_filepath}. Skipping processing for {snapshot_date_str}.")
        return None
    try:
        clickstream_df = spark.read.parquet(clickstream_filepath).withColumnRenamed("snapshot_date", "clickstream_snapshot_date")
        print(f"Loaded Clickstream data from: {clickstream_filepath}, Row count: {clickstream_df.count()}")
    except AnalysisException as e:
        print(f"Error reading Clickstream data: {e}")
        return None

    # Load Silver Attributes Table
    attributes_partition_name = f"silver_feat_attribute_{snapshot_date_str.replace('-', '_')}.parquet"
    attributes_filepath = os.path.join(silver_attributes_directory, attributes_partition_name)
    if not os.path.exists(attributes_filepath):
        print(f"File not found: {attributes_filepath}. Skipping processing for {snapshot_date_str}.")
        return None
    try:
        attributes_df = spark.read.parquet(attributes_filepath).withColumnRenamed("snapshot_date", "attributes_snapshot_date")
        print(f"Loaded Attributes data from: {attributes_filepath}, Row count: {attributes_df.count()}")
    except AnalysisException as e:
        print(f"Error reading Attributes data: {e}")
        return None

    # Load Silver Financial Table
    financial_partition_name = f"silver_feat_financial_{snapshot_date_str.replace('-', '_')}.parquet"
    financial_filepath = os.path.join(silver_financial_directory, financial_partition_name)
    if not os.path.exists(financial_filepath):
        print(f"File not found: {financial_filepath}. Skipping processing for {snapshot_date_str}.")
        return None
    try:
        financial_df = spark.read.parquet(financial_filepath).withColumnRenamed("snapshot_date", "financial_snapshot_date")
        print(f"Loaded Financial data from: {financial_filepath}, Row count: {financial_df.count()}")
    except AnalysisException as e:
        print(f"Error reading Financial data: {e}")
        return None

    # Combine datasets
    gold_table_df = clickstream_df.join(attributes_df, on="Customer_ID", how="inner") \
                                  .join(financial_df, on="Customer_ID", how="inner")

    # Add snapshot_date column
    gold_table_df = gold_table_df.withColumn("snapshot_date", to_date(lit(snapshot_date_str), "yyyy-MM-dd"))

    # Drop unnecessary columns
    gold_table_df = gold_table_df.drop("Monthly_Inhand_Salary", "Outstanding_Debt", "Changed_Credit_Limit")

    # One-hot encode "Payment_of_Min_Amount"
    payment_min_amount_classes = gold_table_df.select("Payment_of_Min_Amount").distinct().rdd.flatMap(lambda x: x).collect()
    for category in payment_min_amount_classes:
        gold_table_df = gold_table_df.withColumn(
            f"Payment_of_Min_Amount_{category}",
            when(col("Payment_of_Min_Amount") == category, 1).otherwise(0)
        )
    gold_table_df = gold_table_df.drop("Payment_of_Min_Amount")

    # One-hot encode "occupation"
    occupation_classes = gold_table_df.select("occupation").distinct().rdd.flatMap(lambda x: x).collect()
    for category in occupation_classes:
        gold_table_df = gold_table_df.withColumn(
            f"occupation_{category.replace(' ', '_')}",
            when(col("occupation") == category, 1).otherwise(0)
        )
    gold_table_df = gold_table_df.drop("occupation")

    # Save the Gold Table
    gold_partition_name = f"gold_table_{snapshot_date_str.replace('-', '_')}.parquet"
    gold_filepath = os.path.join(gold_table_directory, gold_partition_name)
    gold_table_df.write.mode("overwrite").parquet(gold_filepath)

    print(f"Gold table saved to: {gold_filepath}")
    return gold_table_df