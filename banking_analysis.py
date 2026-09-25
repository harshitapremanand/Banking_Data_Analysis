import pandas as pd
import numpy as np


df = pd.read_csv("dataset/bank-full.csv", sep=";")

print("=" * 60)
print("BANKING DATA - DATA CLEANING")
print("=" * 60)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

print("\nUNKNOWN VALUES:")
print((df == "unknown").sum())

categorical_columns = ["job", "education", "contact", "poutcome"]

for column in categorical_columns:
    df[column] = df[column].replace("unknown", "not_available")

print("\nUNKNOWN VALUES AFTER CLEANING:")
print((df == "unknown").sum())

print("\nNUMERIC FIELD VALIDATION:")

print("Invalid age values:", (df["age"] <= 0).sum())
print("Negative balance values:", (df["balance"] < 0).sum())
print("Invalid day values:", ((df["day"] < 1) | (df["day"] > 31)).sum())
print("Invalid duration values:", (df["duration"] < 0).sum())
print("Invalid campaign values:", (df["campaign"] < 1).sum())
print("Invalid previous values:", (df["previous"] < 0).sum())

print("\nCATEGORICAL FIELD VALIDATION:")

print("Marital values:")
print(df["marital"].unique())

print("\nHousing values:")
print(df["housing"].unique())

print("\nLoan values:")
print(df["loan"].unique())

print("\nDefault values:")
print(df["default"].unique())

print("\nTarget values:")
print(df["y"].unique())

numeric_columns = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

categorical_columns = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "y"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

for column in categorical_columns:
    df[column] = df[column].astype("string")

print("\nDATA TYPES AFTER CLEANING:")
print(df.dtypes)

print("\nMISSING VALUES AFTER TYPE CONVERSION:")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("FINAL DATA VALIDATION")
print("=" * 60)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nRemaining Unknown Values:")
print((df == "unknown").sum().sum())

print("\nFinal Dataset Shape:")
print(df.shape)

df = pd.read_csv("dataset/bank-full-cleaned.csv", sep=";")

print("\n" + "=" * 60)
print("DATA TRANSFORMATION")
print("=" * 60)

# Transaction Amount Category
df["Transaction_Amount_Category"] = np.select(
    [
        df["balance"] < 0,
        (df["balance"] >= 0) & (df["balance"] < 1000),
        (df["balance"] >= 1000) & (df["balance"] < 5000),
        df["balance"] >= 5000
    ],
    [
        "Negative",
        "Low",
        "Medium",
        "High"
    ],
    default="Unknown"
)

print("\nTransaction Amount Category:")
print(df["Transaction_Amount_Category"].value_counts())

# Validate Transaction Amount Category

valid_categories = ["Negative", "Low", "Medium", "High"]

invalid_categories = ~df["Transaction_Amount_Category"].isin(valid_categories)

print("\nTransaction Amount Category Validation:")
print("Invalid categories:", invalid_categories.sum())

print("\nCategory counts:")
print(df["Transaction_Amount_Category"].value_counts())

# Account Balance Status

df["Account_Balance_Status"] = np.select(
    [
        df["balance"] < 0,
        (df["balance"] >= 0) & (df["balance"] < 1000),
        (df["balance"] >= 1000) & (df["balance"] < 5000),
        df["balance"] >= 5000
    ],
    [
        "Overdrawn",
        "Low_Balance",
        "Normal_Balance",
        "High_Balance"
    ],
    default="Unknown"
)

print("\nAccount Balance Status:")
print(df["Account_Balance_Status"].value_counts())

valid_balance_status = [
    "Overdrawn",
    "Low_Balance",
    "Normal_Balance",
    "High_Balance"
]

invalid_balance_status = ~df["Account_Balance_Status"].isin(valid_balance_status)

print("\nAccount Balance Status Validation:")
print("Invalid statuses:", invalid_balance_status.sum())

# Loan Indicator

df["Loan_Indicator"] = np.where(
    df["loan"] == "yes",
    1,
    0
)

print("\nLoan Indicator:")
print(df["Loan_Indicator"].value_counts())

# Validate Loan Indicator

invalid_loan_indicator = ~df["Loan_Indicator"].isin([0, 1])

print("\nLoan Indicator Validation:")
print("Invalid indicators:", invalid_loan_indicator.sum())

# Transaction Indicator

df["Transaction_Indicator"] = np.where(
    df["campaign"] > 1,
    1,
    0
)

print("\nTransaction Indicator:")
print(df["Transaction_Indicator"].value_counts())

# Validate Transaction Indicator

invalid_transaction_indicator = ~df["Transaction_Indicator"].isin([0, 1])

print("\nTransaction Indicator Validation:")
print("Invalid indicators:", invalid_transaction_indicator.sum())



print("\n" + "=" * 60)
print("FINAL TRANSFORMATION VALIDATION")
print("=" * 60)

transformed_columns = [
    "Transaction_Amount_Category",
    "Account_Balance_Status",
    "Loan_Indicator",
    "Transaction_Indicator"
]

print("\nMissing values in transformed columns:")
print(df[transformed_columns].isnull().sum())

print("\nNumber of records in transformed columns:")
print(df[transformed_columns].count())

print("\nFinal Dataset Shape:")
print(df.shape)

# ==========================================
# DATA ANALYSIS
# ==========================================

print("\n" + "=" * 60)
print("BANKING DATA ANALYSIS")
print("=" * 60)

# Load cleaned dataset
analysis_df = pd.read_csv(
    "dataset/bank-full-cleaned.csv",
    sep=";"
)

print("\nAnalysis dataset loaded successfully!")
print("Dataset Shape:", analysis_df.shape)

# Total account balance analysis

total_balance = analysis_df["balance"].sum()
average_balance = analysis_df["balance"].mean()
minimum_balance = analysis_df["balance"].min()
maximum_balance = analysis_df["balance"].max()

print("\nACCOUNT BALANCE ANALYSIS:")
print("Total Account Balance:", total_balance)
print("Average Account Balance:", round(average_balance, 2))
print("Minimum Account Balance:", minimum_balance)
print("Maximum Account Balance:", maximum_balance)

# Customer activity by job type

customer_activity = analysis_df["job"].value_counts()

print("\nCUSTOMER ACTIVITY BY JOB TYPE:")
print(customer_activity)

# Loan information analysis

loan_counts = analysis_df["loan"].value_counts()

print("\nLOAN INFORMATION:")
print(loan_counts)

print("\nLoan Percentages:")
print((analysis_df["loan"].value_counts(normalize=True) * 100).round(2))

# Housing loan analysis

housing_counts = analysis_df["housing"].value_counts()

print("\nHOUSING LOAN INFORMATION:")
print(housing_counts)

print("\nHousing Loan Percentages:")
print((analysis_df["housing"].value_counts(normalize=True) * 100).round(2))

# Customer contact analysis

contact_counts = analysis_df["contact"].value_counts()

print("\nCUSTOMER CONTACT ANALYSIS:")
print(contact_counts)

print("\nContact Percentages:")
print((analysis_df["contact"].value_counts(normalize=True) * 100).round(2))

# Campaign outcome analysis

campaign_outcome = analysis_df["y"].value_counts()

print("\nCAMPAIGN OUTCOME:")
print(campaign_outcome)

print("\nCampaign Outcome Percentages:")
print((analysis_df["y"].value_counts(normalize=True) * 100).round(2))

marital_activity = analysis_df["marital"].value_counts()

print("\nCUSTOMER ACTIVITY BY MARITAL STATUS:")
print(marital_activity)

print("\nMarital Status Percentages:")
print((analysis_df["marital"].value_counts(normalize=True) * 100).round(2))


education_activity = analysis_df["education"].value_counts()

print("\nCUSTOMER ACTIVITY BY EDUCATION:")
print(education_activity)

print("\nEducation Percentages:")
print((analysis_df["education"].value_counts(normalize=True) * 100).round(2))

analysis_df["Age_Group"] = np.select(
    [
        analysis_df["age"] < 30,
        analysis_df["age"].between(30, 49),
        analysis_df["age"].between(50, 64),
        analysis_df["age"] >= 65
    ],
    [
        "Young",
        "Adult",
        "Senior",
        "Elderly"
    ],
    default="Unknown"
)

age_group_counts = analysis_df["Age_Group"].value_counts()

print("\nCUSTOMER ACTIVITY BY AGE GROUP:")
print(age_group_counts)


average_balance_by_job = (
    analysis_df.groupby("job")["balance"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAVERAGE ACCOUNT BALANCE BY JOB:")
print(average_balance_by_job.round(2))

loan_campaign = pd.crosstab(
    analysis_df["loan"],
    analysis_df["y"]
)

print("\nLOAN STATUS VS CAMPAIGN OUTCOME:")
print(loan_campaign)

print("\n" + "=" * 60)
print("FINAL ANALYSIS VALIDATION")
print("=" * 60)

print("\nDataset Shape:")
print(analysis_df.shape)

print("\nMissing Values:")
print(analysis_df.isnull().sum())

print("\nDuplicate Rows:")
print(analysis_df.duplicated().sum())

# ==========================================
# LOAD - SAVE PROCESSED DATASET
# ==========================================

output_file = "dataset/bank-full-cleaned.csv"

df.to_csv(output_file, sep=";", index=False)

print("\n" + "=" * 60)
print("ETL LOAD")
print("=" * 60)
print("Processed dataset saved successfully!")
print("Output file:", output_file)
print("Records loaded:", len(df))
