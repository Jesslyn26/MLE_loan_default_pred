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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/lms_loan_daily.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df

def process_bronze_table_feat_attributes(snapshot_date_str, bronze_feat_attributes_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/features_attributes.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_feat_attribute_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_feat_attributes_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df


def process_bronze_table_feat_clickstream(snapshot_date_str, bronze_feat_attributes_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Path to the clickstream CSV file
    csv_file_path = "data/feature_clickstream.csv"

    # Load data and filter by snapshot_date
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + ' row count:', df.count())

    # Save Bronze table to datamart
    partition_name = "bronze_feat_clickstream_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = bronze_feat_attributes_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('Saved to:', filepath)

    return df


def process_bronze_table_feat_financial(snapshot_date_str, bronze_feat_attributes_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Path to the clickstream CSV file
    csv_file_path = "data/features_financials.csv"

    # Load data and filter by snapshot_date
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + ' row count:', df.count())

    # Save Bronze table to datamart
    partition_name = "bronze_feat_financial_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = bronze_feat_attributes_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('Saved to:', filepath)

    return df
