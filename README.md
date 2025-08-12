Project Title: Catalog Uplift Modeling – Project Workflow

1.Problem Statement:

Develop a data-driven targeting approach for catalog marketing that:
- Identifies customers who are truly incremental purchasers as a result of receiving a catalog, rather than purchasers who would have bought anyway.
- Estimates the size of this incremental audience and produces a ranked list of customers prioritized by expected incremental lift.
- Focuses on customers who need a catalog to purchase incrementally, explicitly distinguishing them from customers whose purchases merely coincide with catalog sends.


 2.Data Collection.
We collected DM (customer master) and transaction data from the SSMS server and ingested them into the Lakehouse via the **kartheek_pipeline.** We then began data exploration to understand the spread of the data (distributions, ranges, counts, and outliers) across key fields before further modeling.

Data Exploration (DM Data)

| Column Name            | Description |
|------------------------|----------------------------------------------------------------|
| Cust_ID                | Unique identifier for the customer in the system. |
| Src_Sys_Nm             | Source system name from which the customer record originates (e.g., MCD). |
| Gold_Cust_ID           | Master or consolidated customer ID after deduplication. |
| Cust_Combine_Id        | Combined key with source prefix and customer ID for uniqueness. |
| Zip5                   | 5-digit ZIP code of the customer's address. |
| nt_call_ct             | Count of "NT" (possibly non-transactional) calls made to the customer. |
| rtl_call_rec           | Indicates whether retail calls were recorded (Y/N). |
| business_ind           | Indicator showing if the customer is a business account (e.g., RB2B). |
| customer_type          | Classification of the customer type (e.g., RB2B = Retail B2B). |
| tradearea_store        | Trade area store associated with the customer. |
| tradearea_store_dist   | Distance from customer's location to the trade area store. |
| Advantage              | Flag indicating participation in the "Advantage" program (Y/N). |
| MP_GAN                 | Marketing Program Global Account Number — links to campaign targeting. |
| Mailed_Date            | Date the marketing material or offer was mailed to the customer. |
| MPMAIL_IND             | Indicator if the customer was mailed (blank, 1, or 0). |
| MP_MKY                 | Marketing Program key — campaign identifier. |
| MP_CAT                 | Marketing program category code. |
| Managed                | Indicates if the customer is under managed account handling (Y/N). |
| MP_CAT_TYPE            | Description of marketing program category (e.g., Advantage). |
| Group_Flag             | Grouping flag used for campaign segmentation (e.g., Mailed). |
| Post_Sales             | Sales amount after the campaign mailing. |
| Post_Margin            | Margin amount after the campaign mailing. |
| Post_Units             | Units purchased after the campaign mailing. |
| Post_Transactions      | Number of transactions after the campaign mailing. |
| row_num                | Row sequence number (useful for deduplication logic). |
| PreDM_Sales            | Sales amount before the Direct Marketing (DM) campaign. |
| PreDM_Margin           | Margin amount before the DM campaign. |
| PreDM_Units            | Units purchased before the DM campaign. |
| PreDM_Transactions     | Number of transactions before the DM campaign. |




Column Descriptions (Sales & Transaction)

| Column Name            | Description |
|------------------------|-------------|
| Gold_Cust_ID           | Unique customer ID used across systems. |
| Tran_Id                | Unique transaction identifier. |
| tran_dt                | Date of transaction (format: DD-MM-YYYY). |
| Item_Sk                | Item SKU (Stock Keeping Unit) or product code. |
| Sle_Qty                | Quantity sold in the transaction. |
| checkout_sales         | Final sales amount paid at checkout. |
| Product_Margin         | Gross profit margin on product (before logistics, shipping, etc.). |
| Landed_Margin          | Net profit including shipping and handling costs. |
| quantity               | Units of product purchased (may match `Sle_Qty`). |
| Prc_Gross_Amt          | Gross price before any discounts. |
| Prc_Offer_Amt          | Discounted offer price (if applicable). |
| Prc_CheckOut_Amt       | Actual price paid by the customer. |
| Dir_Ord_Chan_Nm        | Order channel name (e.g., online, retail). |
| OnlineStore_Cd         | Online store code or ID. |
| SameDayCancel_flg      | Flag if the order was cancelled on the same day (1 = Yes, 0 = No). |
| Quote_Release_Dt_SK    | Quote release date surrogate key (if any). |
| BOPIS_Flg              | Buy Online, Pick-up In Store flag (1 = Yes). |
| Cust_Addr_Available_Flg| Flag indicating if customer address is available. |
| Brand                  | Brand name of the product. |
| Subclass               | Product subcategory. |
| vendor_nm              | Name of the vendor or manufacturer. |
| standard_cost          | Standard procurement cost to the company. |
| class                  | Product category (e.g., warranty plans, electronics). |
| class_code             | Numeric/class code for the product category. |
| subclass_code          | Numeric/subclass code for the subcategory. |




### Installing All The Requirements:
									
													
%pip install imbalanced-learn


%pip install imbalanced-learn is used to install the imbalanced-learn library, which provides tools to handle **imbalanced datasets**  (e.g., oversampling with SMOTE, undersampling, etc.) directly in your Python/Spark environment.



# --- Import all packages required---
import re
import os
import math
import warnings
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.sql.functions import col, when, ntile
from pyspark.sql.window import Window
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from flaml.automl.spark.utils import to_pandas_on_spark
import mlflow
from mlflow.tracking import MlflowClient
import flaml
from flaml import AutoML

-----------------------

### Data Loading (Catalog Data)

# Load the data from the specified table in the catalog into a Spark DataFrame
df_dm = spark.sql("SELECT * FROM Kartheek_Pipeline.Catalog_Data")

# Display the DataFrame in a tabular view for quick inspection
display(df_dm)

# Get the list of column names in the DataFrame
df_dm.columns

### Code Explanation:

This code reads the Catalog_Data table into Spark, shows the data for review, and lists all its column names for reference.

------------------------


### Data Loading (Sales)

# Parameters Setup

# Name of the database table containing all customers' sales data
db_sales = "Kartheek_Pipeline.sales_allcustomers"

# Get today's date in YYYYMMDD format to append to table names (ensures unique table names each day)
today_str = datetime.today().strftime('%Y%m%d')

# Name for the scored output table (stores predicted propensity scores)
scored_table_name = f"salesData.Scored_Propensity_{today_str}"

# Name for the clustered output table (stores clustered customer segments)
clustered_table_name = f"salesData.ScoredClusteredAudience_{today_str}"


# -------------------------------------
# Load Sales Data

# Read the sales data from the specified database table into a Spark DataFrame
df_sales = spark.sql(f"SELECT * FROM {db_sales}")

# Display the loaded DataFrame in the notebook for inspection
display(df_sales)

# Retrieve and display the list of column names from the sales DataFrame
df_sales.columns

### Code Explanation:

db_sales: Points to your sales data source table in your Fabric catalog.

today_str: Adds a timestamp to output tables so each day's run produces separate tables without overwriting.

scored_table_name & clustered_table_name: Used later to save results of your ML scoring and clustering steps.

spark.sql(...): Runs a SQL query against your Spark SQL catalog to load data.

display(df_sales): Shows the first few rows in a table-like format for quick visual verification.

df_sales.columns: Returns all column names, helpful for checking schema before further processing.

---------------------------------
### 3. Simple Exploratory Data Analysis:


from pyspark.sql import functions as F

def simple_eda(df):
    # 1. Shape info
    print("=== Shape ===")
    print(f"Rows: {df.count()}, Columns: {len(df.columns)}\n")
    
    # 2. Schema & data types
    print("=== Schema & Data Types ===")
    df.printSchema()
    print()
    
    # 3. Null counts per column
    print("=== Null Counts per Column ===")
    null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
    null_counts.show(vertical=True)
    
    # 4. Distinct counts per column
    print("=== Distinct Counts per Column ===")
    distinct_counts = df.select([F.countDistinct(c).alias(c) for c in df.columns])
    distinct_counts.show(vertical=True)
    
    # 5. Quick sample
    print("=== Sample Data ===")
    df.show(5, truncate=False)

# Return the Function 
simple_eda(df_valid)

### Code Explanation:

- Shape Information – Shows how many rows and columns you have. df.count() triggers a full scan, so it can be slow on big datasets. Counting columns is fast.

- Schema & Data Types – Lets you check if Spark inferred the correct types and spot mistakes like numbers stored as strings.

- Null Counts – Tells you how many missing values each column has, helping you decide on imputation, dropping, or other cleaning steps.

- Distinct Counts – Shows how many unique values each column contains, useful for identifying categorical variables, continuous features, or unique IDs.

- Sample Data – Displays a few records so you can visually inspect the structure and content.

 ---------------------------

# Print all column names in the DataFrame for inspection
print(df_valid.columns)

# Count the number of unique customers based on 'Gold_Cust_ID'
num_customers = df_valid.select("Gold_Cust_ID").distinct().count()

# Display the number of distinct customers in the DataFrame
print(f"Number of distinct customers in df_scored: {num_customers}")

### Code Explanation

1. `print(df_valid.columns)`  
   - Prints all column names in the DataFrame `df_valid` as a Python list.  
   - Useful for quickly checking available fields.

2. `df_valid.select("Gold_Cust_ID")`  
   - Selects only the `Gold_Cust_ID` column from `df_valid`.

3. `.distinct()`  
   - Removes duplicate `Gold_Cust_ID` values, keeping only unique customer IDs.

4. `.count()`  
   - Counts the number of unique customers.

5. `num_customers = ...`  
   - Stores the count of unique customers in the variable `num_customers`.

6. `print(f" Number of distinct customers in df_scored: {num_customers}")`  
   - Prints the count of unique customers with a descriptive message.

-------------------------------------

# Save df_valid as a managed table
df_valid.write.mode("overwrite").saveAsTable("Kartheek_Pipeline.Catalog_Universe_with_Sales")


### Code Explanation
- **Purpose:** Saves the DataFrame `df_valid` as a **managed Delta table** in the metastore.  
- **`write.mode("overwrite")`** → Overwrites the table if it already exists.  
- **`saveAsTable("Kartheek_Pipeline.Catalog_Universe_with_Sales")`** → Creates or replaces a managed table with the given name in the `Kartheek_Pipeline` schema.

-------------------------------------------------------------------


### 4.Feature Engineering – Catalog Segmentation

####  Creating the tables For B2C (Catalog Consumers)

### Code Explanation
1. `df = spark.sql("SELECT * FROM Kartheek_Pipeline.catalog_universe_with_sales")`  
   - Runs a Spark SQL query to select all records from the `Kartheek_Pipeline.catalog_universe_with_sales` table.  
   - Loads the result into a DataFrame called `df`.

----------------------------------------------------


# Filtering the Cons Column 
from pyspark.sql.functions import col, lower
df_cat = df.filter(col("MP_CAT_TYPE") == "catalog")
df_cons = df_cat.filter(lower(col("customer_type")) == "cons")
display(df_cons)

### Code Explanation – Filtering Consumers from Sales Universe Data
- First, the data is filtered to include only records where the mailing type is **Catalog**.  
- Then, from this subset, only **consumer (B2C)** customers are selected based on the `customer_type` column.  
- The result is a dataset containing only catalog records for consumer customers.

-----------------------------------------------
from pyspark.sql import functions as F

# Group the DataFrame by customer, transaction, and mailing-related columns
grouped_df = df.groupBy(
    "Gold_Cust_ID", "Cust_ID", "Src_Sys_Nm", "Cust_Combine_Id", "Zip5",
    "nt_call_ct", "rtl_call_rec", "business_ind", "customer_type",
    "tradearea_store", "tradearea_store_dist", "Advantage", "MP_GAN",
    "Mailed_Date", "MPMAIL_IND", "MP_MKY", "MP_CAT", "Managed",
    "MP_CAT_TYPE", "Group_Flag", "Post_Sales", "Post_Margin",
    "Post_Units", "Post_Transactions", "row_num",
    "PreDM_Sales", "PreDM_Margin", "PreDM_Units", "PreDM_Transactions"
).agg(
    # Sum of numeric metrics
    F.sum("Sle_Qty").alias("Total_Sle_Qty"),                     # Total sales quantity
    F.sum("checkout_sales").alias("Total_Checkout_Sales"),       # Total checkout sales
    F.sum("Product_Margin").alias("Total_Product_Margin"),       # Total product margin
    F.sum("Landed_Margin").alias("Total_Landed_Margin"),         # Total landed margin
    F.sum("quantity").alias("Total_Quantity"),                   # Total quantity sold

    # Average of price-related fields
    F.avg("Prc_Gross_Amt").alias("Avg_Prc_Gross_Amt"),            # Average gross price
    F.avg("Prc_Offer_Amt").alias("Avg_Prc_Offer_Amt"),            # Average offer price
    F.avg("Prc_CheckOut_Amt").alias("Avg_Prc_CheckOut_Amt"),      # Average checkout price

    # Distinct counts
    F.countDistinct("Dir_Ord_Chan_Nm").alias("Distinct_Channels"),# Number of unique order channels

    # Unique value collections
    F.collect_set("Brand").alias("Unique_Brands"),                # Set of unique brands purchased
    F.collect_set("Subclass").alias("Unique_Subclasses"),         # Set of unique subclasses
    F.collect_set("vendor_nm").alias("Vendors"),                  # Set of unique vendors

    # Average cost
    F.avg("standard_cost").alias("Avg_Standard_Cost"),            # Average standard cost

    # First value pick (non-null)
    F.first("OnlineStore_Cd").alias("First_OnlineStore_Cd"),      # First online store code
    F.first("SameDayCancel_flg").alias("First_SameDayCancel_flg"),# First same-day cancel flag
    F.first("Quote_Release_Dt_SK").alias("First_QuoteReleaseDt"), # First quote release date
    F.first("BOPIS_Flg").alias("First_BOPIS_Flg"),                 # First BOPIS flag
    F.first("Cust_Addr_Available_Flg").alias("First_Cust_Addr_Avail") # First customer address available flag
)


### Code Explanation – Grouping and Aggregation
- **Purpose:** Summarize transaction and customer data by grouping on customer, store, and mailing-related fields.
- **Aggregations performed:**
  - **Sum:** Calculates total sales quantities, checkout sales, product margins, landed margins, and total quantity.
  - **Average:** Computes average prices (gross, offer, checkout) and average standard cost.
  - **Distinct Count:** Counts unique order channels per group.
  - **Collect Set:** Gathers unique brands, subclasses, and vendors into lists.
  - **First Value:** Picks the first recorded value for certain flags or codes.
- **Output:** A condensed DataFrame where each row represents one group with aggregated metrics and unique value sets.

----------------------------------

from pyspark.sql.functions import (
    to_date, col, datediff, mean, stddev, countDistinct, sum, max, first
)

# 1. Ensure date formats
df_cons = df_cons.withColumn("tran_dt", to_date("tran_dt", "dd-MM-yyyy"))
df_cons = df_cons.withColumn("Mailed_Date", to_date("Mailed_Date", "dd-MM-yyyy"))

# 2. Filter only transactions that occurred BEFORE the mail date
df_pre_mail = df_cons.filter(col("tran_dt") < col("Mailed_Date"))

# 3. Aggregate RFM features per customer
rfm_df = df_pre_mail.groupBy("Gold_Cust_ID").agg(
    max("tran_dt").alias("last_transaction_date"),
    countDistinct("Tran_Id").alias("Frequency"),
    sum("checkout_sales").alias("Monetary"),
    first("Mailed_Date").alias("Mailed_Date")  # Reference point for Recency
)

# 4. Compute Recency in days (Mail_Date - Last_Transaction)
rfm_df = rfm_df.withColumn("Recency", datediff(col("Mailed_Date"), col("last_transaction_date")))

# 5. Compute mean and stddev for each RFM column
stats = rfm_df.select(
    mean("Recency").alias("mean_rec"),
    stddev("Recency").alias("std_rec"),
    mean("Frequency").alias("mean_freq"),
    stddev("Frequency").alias("std_freq"),
    mean("Monetary").alias("mean_mon"),
    stddev("Monetary").alias("std_mon")
).collect()[0]

# 6. Broadcast stats
mean_rec, std_rec = stats["mean_rec"], stats["std_rec"]
mean_freq, std_freq = stats["mean_freq"], stats["std_freq"]
mean_mon, std_mon = stats["mean_mon"], stats["std_mon"]

# 7. Add standardized z-scores
rfm_df = rfm_df \
    .withColumn("Recency_Z", (mean_rec - col("Recency")) / std_rec) \
    .withColumn("Frequency_Z", (col("Frequency") - mean_freq) / std_freq) \
    .withColumn("Monetary_Z", (col("Monetary") - mean_mon) / std_mon) \
    .withColumn("RFM_Z_Composite", col("Recency_Z") + col("Frequency_Z") + col("Monetary_Z"))

# 8. (Optional) View sample
rfm_df.select("Gold_Cust_ID", "Recency", "Frequency", "Monetary", 
              "Recency_Z", "Frequency_Z", "Monetary_Z", "RFM_Z_Composite").show(10)



### Code Explanation – RFM Analysis

1. **Convert Dates**  
   - Ensure `tran_dt` and `Mailed_Date` are in proper date format.

2. **Filter Pre-Mail Transactions**  
   - Keep only transactions that happened **before** the mail date.

3. **Aggregate RFM Features**  
   - For each customer:
     - `last_transaction_date` → Most recent transaction date before mail.  
     - `Frequency` → Number of distinct transactions.  
     - `Monetary` → Total sales before mail.  
     - `Mailed_Date` → Reference date for recency calculation.

4. **Calculate Recency**  
   - Days between mail date and last transaction.

5. **Compute Stats**  
   - Get mean and standard deviation for Recency, Frequency, and Monetary.

6. **Standardize (Z-Scores)**  
   - Create normalized scores for each RFM metric.  
   - Combine them into `RFM_Z_Composite` for an overall score.

7. **View Sample Output**  
   - Show first 10 customers with raw and standardized RFM metrics.


-----------------------------------


# final_df will have all features + RFM z-scores
final_df = grouped_df.join(
    rfm_df.select("Gold_Cust_ID", "Recency", "Frequency", "Monetary",
                  "Recency_Z", "Frequency_Z", "Monetary_Z", "RFM_Z_Composite"),
    on="Gold_Cust_ID",
    how="left"
)


### Code Explanation - Joining RFM Scores to Main Dataset

- **Purpose**:  
  Add RFM metrics (`Recency`, `Frequency`, `Monetary`) and their standardized z-scores to the main customer dataset.

- **Logic**:  
  - Match records from `grouped_df` with `rfm_df` using `Gold_Cust_ID`.
  - Keep all rows from `grouped_df` (left join).
  - Bring over only the required RFM columns from `rfm_df`.

- **Result**:  
  `final_df` now contains both:
    1. Original features from `grouped_df`.
    2. RFM-related numeric features for modeling or segmentation.


----------------------------------


from pyspark.sql.functions import col, sqrt, pow, lit, when

# Step 1: Normalize Pre and Post features
final_df = final_df.withColumns({
    "PreDM_Sales_Rate": col("PreDM_Sales") / 90,
    "PreDM_Margin_Rate": col("PreDM_Margin") / 90,
    "PreDM_Units_Rate": col("PreDM_Units") / 90,
    "PreDM_Transactions_Rate": col("PreDM_Transactions") / 90,

    "Post_Sales_Rate": col("Post_Sales") / 30,
    "Post_Margin_Rate": col("Post_Margin") / 30,
    "Post_Units_Rate": col("Post_Units") / 30,
    "Post_Transactions_Rate": col("Post_Transactions") / 30
})

# Step 2: Add SpendRate_Z and Response Flag
epsilon = 1e-6

final_df = final_df.withColumn(
    "SpendRate_Z", 
    (col("Post_Sales_Rate") - col("PreDM_Sales_Rate")) /
    sqrt(pow(col("PreDM_Margin_Rate"), 2) + pow(col("Post_Margin_Rate"), 2) + lit(epsilon))
).withColumn(
    "response_flag", 
    when((col("SpendRate_Z") > 1.645) & (col("Group_Flag") == "Mailed"), 1).otherwise(0)
)

### Step 1: Normalize Pre and Post Campaign Metrics
- Creates **rate features** for both:
  - **Pre-campaign** (90 days)
  - **Post-campaign** (30 days)
- Instead of using raw totals (e.g., total sales),  
  calculates **daily averages** by dividing totals by the number of days in each period.
- This makes metrics **comparable** across periods, even if their time windows differ.

### Step 2: Calculate `SpendRate_Z` and Response Flag
- **SpendRate_Z**:
  - Measures how much the **post-campaign sales rate** differs from the **pre-campaign sales rate**.
  - Adjusted for variability in margins (standard deviation).
  - Higher values = greater positive change in spending after the campaign.
  
- **response_flag**:
  - Binary indicator of campaign influence.
  - **1** → Customer had a significant positive change (`SpendRate_Z > 1.645`) **and** was part of the "Mailed" group.
  - **0** → Otherwise.
- Essentially flags customers **likely influenced** by the campaign..







