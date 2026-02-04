"""
Preprocessing script for iPhone Review Sentiment Analysis

Input:
- Raw CSV: ../data/iphone.csv

Output:
- Train CSV: ../data/processed/train.csv
- Validation CSV: ../data/processed/validation.csv

Columns produced:
- text  : review text
- label : binary sentiment label (0 = negative, 1 = positive)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------
# CONFIG
# ---------------------------
RAW_DATA_PATH = "../data/iphone.csv"
OUTPUT_DIR = "../data/processed"
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
VAL_PATH = os.path.join(OUTPUT_DIR, "validation.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------
# LOAD DATA
# ---------------------------
print("Loading raw dataset...")
df = pd.read_csv(RAW_DATA_PATH)

print("Available columns:")
print(df.columns.tolist())


# ---------------------------
# SELECT & RENAME COLUMNS
# ---------------------------
# reviewDescription → text
# ratingScore      → label
df = df[["reviewDescription", "ratingScore"]]

df = df.rename(columns={
    "reviewDescription": "text",
    "ratingScore": "label"
})


# ---------------------------
# CLEANING
# ---------------------------
print("Cleaning data...")

# Drop missing text or labels
df = df.dropna(subset=["text", "label"])

# Convert ratings to numeric
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])

# Convert star ratings to binary sentiment
# 1–2 → Negative (0)
# 4–5 → Positive (1)
# Drop neutral (3)
def rating_to_binary(rating):
    if rating <= 2:
        return 0
    elif rating >= 4:
        return 1
    else:
        return None

df["label"] = df["label"].apply(rating_to_binary)
df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)


# ---------------------------
# TRAIN / VALIDATION SPLIT
# ---------------------------
print("Splitting train / validation...")

train_df, val_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)


# ---------------------------
# SAVE OUTPUT
# ---------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)

print("\nPreprocessing complete ✅")
print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Saved to: {OUTPUT_DIR}")
