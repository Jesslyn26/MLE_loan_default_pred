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

from pyspark.sql.functions import col, to_date, when, mean, round, regexp_extract, regexp_replace,  min, date_add, lit, year, month, split
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, StructField, StructType
from pyspark.sql.utils import AnalysisException
from pyspark.sql import Window
import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_silver_table_feat_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):

    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to Bronze table
    partition_name = f"bronze_feat_attribute_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_attributes_directory, partition_name)
    
    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. Skipping processing for {snapshot_date_str}.")
        return None

    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('Loaded from:', filepath, 'Row count:', df.count())

    # Clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Age": IntegerType(),
        "snapshot_date": DateType(),
        "occupation": StringType(),
        "SSN": StringType(),
        "Name": StringType()
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Transform snapshot_date column to date
    df = df.withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))

    # Replace "_______" in the occupation column with None
    df = df.withColumn(
        "occupation",
        when(col("occupation") == "_______", None).otherwise(col("occupation"))
    )

    # Calculate the mean of valid age values (age >= 0 and age <= 120)
    mean_age = df.filter((col("Age") >= 0) & (col("Age") <= 120)) \
        .select(mean(col("Age")).alias("mean_age")) \
        .collect()[0]["mean_age"]

    # Replace invalid age values (age < 0, age > 120, or NULL) with the mean age
    df = df.withColumn(
    "Age",
    when((col("Age") < 0) | (col("Age") > 120) | (col("Age").isNull()), mean_age).otherwise(col("Age"))
)

    df = df.withColumn("Age", col("Age").cast("int"))

    # Drop unnecessary columns
    df = df.drop("SSN", "Name")

    # Replace NULL values in occupation with "Teacher"
    df = df.withColumn(
        "occupation",
        when(col("occupation").isNull(), "Teacher").otherwise(col("occupation"))
    )

    # Save Silver table
    partition_name = f"silver_feat_attribute_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_attributes_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print('Saved to:', filepath)

    return df


def process_silver_table_feat_financial(snapshot_date_str, bronze_financial_directory, silver_financial_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to Bronze table
    partition_name = f"bronze_feat_financial_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_financial_directory, partition_name)
    
    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. Skipping processing for {snapshot_date_str}.")
        return None

    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('Loaded from:', filepath, 'Row count:', df.count())

    # Clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": IntegerType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": FloatType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": IntegerType(),
        "snapshot_date": DateType()
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Transform snapshot_date column to date
    df = df.withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))

    # Replace negative values with NULL
    df = df.withColumn(
        "Num_of_Loan", when(col("Num_of_Loan") < 0, None).otherwise(col("Num_of_Loan"))
    ).withColumn(
        "Num_of_Delayed_Payment", when(col("Num_of_Delayed_Payment") < 0, None).otherwise(col("Num_of_Delayed_Payment"))
    ).withColumn(
        "Changed_Credit_Limit", when(col("Changed_Credit_Limit") < 0, None).otherwise(col("Num_of_Delayed_Payment"))
    )

    # Impute missing values with medians
    median_dict = {
        "Annual_Income": df.approxQuantile("Annual_Income", [0.5], 0.01)[0],
        "Outstanding_Debt": df.approxQuantile("Outstanding_Debt", [0.5], 0.01)[0],
        "Amount_invested_monthly": df.approxQuantile("Amount_invested_monthly", [0.5], 0.01)[0],
        "Num_of_Loan": df.approxQuantile("Num_of_Loan", [0.5], 0.01)[0],
        "Num_of_Delayed_Payment": df.approxQuantile("Num_of_Delayed_Payment", [0.5], 0.01)[0],
        "Changed_Credit_Limit": df.approxQuantile("Changed_Credit_Limit", [0.5], 0.01)[0]
    }
    df = df.fillna(median_dict)

    # Fill missing Monthly_Balance with mean
    monthly_balance_mean = df.select(mean(col("Monthly_Balance"))).collect()[0][0]
    df = df.fillna({"Monthly_Balance": monthly_balance_mean})

    # Clean and transform Type_of_Loan
    df = df.withColumn(
    "Type_of_Loan", when(col("Type_of_Loan").isNull(), "Not Specified").otherwise(col("Type_of_Loan"))
    ).withColumn(
        "Type_of_Loan", regexp_replace(col("Type_of_Loan"), r"\s+and\s+", ", ")
    ).withColumn(
        "Type_of_Loan", regexp_replace(col("Type_of_Loan"), r",\s*,", ",")
    ).withColumn(
        "Type_of_Loan", regexp_replace(col("Type_of_Loan"), r",\s*$", "")
    )
    
    # Split Type_of_Loan into an array
    df = df.withColumn("Type_of_Loan_Array", split(col("Type_of_Loan"), r",\s*"))
    
    # Drop the original Type_of_Loan column
    df = df.drop("Type_of_Loan")

    # Replace "_" with "Unknown" in Credit_Mix
    df = df.withColumn(
        "Credit_Mix", when(col("Credit_Mix") == "_", "Unknown").otherwise(col("Credit_Mix"))
    )

    # Replace invalid Payment_Behaviour values with "Unknown"
    df = df.withColumn(
        "Payment_Behaviour",
        when(col("Payment_Behaviour").rlike(r"^[a-zA-Z_]+$"), col("Payment_Behaviour")).otherwise("Unknown")
    )

    # Transform Credit_History_Age to years
    df = df.withColumn(
        "Credit_History_Age_Years",
        (
            regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years", 1).cast("int") +
            (regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months", 1).cast("int") / 12)
        )
    ).drop("Credit_History_Age")

    # Save Silver table
    partition_name = f"silver_feat_financial_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_financial_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print('Saved to:', filepath)

    return df


def process_silver_table_feat_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Connect to Bronze table
    partition_name = f"bronze_feat_clickstream_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_clickstream_directory, partition_name)

    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. Skipping processing for {snapshot_date_str}.")
        return None

    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('Loaded from:', filepath, 'Row count:', df.count())

    # Clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }
    # Add all `fe_*` columns as FloatType
    for i in range(1, 21):  # Assuming fe_1 to fe_20
        column_type_map[f"fe_{i}"] = FloatType()

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Transform snapshot_date column to date
    df = df.withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))

    # Step 1: Find the earliest snapshot_date for each Customer_ID
    earliest_date_df = df.groupBy("Customer_ID").agg(min("snapshot_date").alias("earliest_snapshot_date"))

    # Step 2: Join the earliest snapshot_date back to the original DataFrame
    df = df.join(earliest_date_df, on="Customer_ID", how="inner")

    # Step 3: Calculate the difference in months from the earliest snapshot_date
    df = df.withColumn(
        "months_since_joining",
        ((year(col("snapshot_date")) - year(col("earliest_snapshot_date"))) * 12 +
         (month(col("snapshot_date")) - month(col("earliest_snapshot_date"))))
    )

    # Step 4: Filter to keep only the first 6 months since joining
    df = df.filter(col("months_since_joining") < 6)

    # Step 5: Drop unnecessary columns
    df = df.drop("earliest_snapshot_date", "months_since_joining")

    # Normalize 'fe_*' columns
    fe_columns = [f"fe_{i}" for i in range(1, 21)]  # Assuming fe_1 to fe_20
    bronze_clickstream_pd = df.toPandas()

    # Apply StandardScaler to normalize 'fe_*' columns
    scaler = StandardScaler()
    bronze_clickstream_pd[fe_columns] = scaler.fit_transform(bronze_clickstream_pd[fe_columns])

    # Convert back to PySpark DataFrame
    df = spark.createDataFrame(bronze_clickstream_pd)

    # Save Silver table
    partition_name = f"silver_feat_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_clickstream_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print('Saved to:', filepath)

    return df