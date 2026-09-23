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