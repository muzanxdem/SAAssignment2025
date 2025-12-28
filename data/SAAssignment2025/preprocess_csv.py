import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import kagglehub
import os

print("--- Starting Data Preprocessing ---")
start_time = time.time()

path = "SA-csv/"

# Download dataset from Kaggle (downloads to default cache location)
# kagglehub.dataset_download("chethuhn/network-intrusion-dataset", path=path)
# curl -L -o ~/DL4IDS/data/CICIDS2017/csv/network-intrusion-dataset.zip  https://www.kaggle.com/api/v1/datasets/download/chethuhn/network-intrusion-dataset

#print("Downloaded to cache:", path)

df = pd.DataFrame()

for file in os.listdir(path):
    if file.endswith(".csv"):
        df_temp = pd.read_csv(os.path.join(path, file))
        df = pd.concat([df, df_temp])
        print(f"Successfully loaded {df.shape[0]} rows from {file}.")
    else:
        print(f"No CSV files found in {path}.")



# Check for NaN values
nan_counts = df.isna().sum()
problematic_nan_cols = nan_counts[nan_counts > 0]

if not problematic_nan_cols.empty:
    print("\nFound columns with NaN values:")
    for col in problematic_nan_cols.index:
        print(f"  {col}: {int(problematic_nan_cols[col])} NaN values")
    print(f"Total NaN values: {nan_counts.sum()}")
    
    # Fill NaN values with the mean of each column (for numeric columns)
    # For non-numeric columns, fill with 0 or the most frequent value
    for col in problematic_nan_cols.index:
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # For numeric columns, fill with mean
            mean_val = df[col].mean()
            if pd.isna(mean_val):
                # If mean is also NaN (all values are NaN), use 0
                df[col].fillna(0, inplace=True)
                print(f"  Column '{col}': Filled {int(nan_counts[col])} NaN values with 0 (mean was also NaN)")
            else:
                df[col].fillna(mean_val, inplace=True)
                print(f"  Column '{col}': Filled {int(nan_counts[col])} NaN values with mean ({mean_val:.4f})")
        else:
            # For non-numeric columns, fill with the most frequent value or empty string
            most_frequent = df[col].mode()
            if len(most_frequent) > 0:
                fill_value = most_frequent.iloc[0]
                df[col].fillna(fill_value, inplace=True)
                print(f"  Column '{col}': Filled {int(nan_counts[col])} NaN values with most frequent value ('{fill_value}')")
            else:
                # If no mode exists (all NaN), fill with empty string
                df[col].fillna('', inplace=True)
                print(f"  Column '{col}': Filled {int(nan_counts[col])} NaN values with empty string (no mode available)")
        
    
    # Verify NaN are fixed
    remaining_nan = df.isna().sum().sum()
    if remaining_nan > 0:
        print(f"WARNING: {remaining_nan} NaN values still remain after filling!")
    else:
        print("All NaN values have been filled.")
else:
    print("No NaN values found in the dataset.")

# Check for infinity values
inf_counts = df.isin([np.inf, -np.inf]).sum()
problematic_inf_cols = inf_counts[inf_counts > 0]

if not problematic_inf_cols.empty:
    print("\nFound columns with infinity values:")
    for col in problematic_inf_cols.index:
        print(f"  {col}: {int(problematic_inf_cols[col])} infinity values")
    
    # Replace all infinity values (-inf and inf) with 0. You can also use df.mean() for a different strategy.
    df.replace([np.inf, -np.inf], 0, inplace=True)
    print("All infinity values have been replaced with 0.")
else:
    print("No infinity values found in the dataset.")



# Separate features and labels. The last column is the label, all others are features. 
# This is depending on the dataset.
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convert all non-numeric features to numeric
print("\n--- Converting non-numeric features to numeric ---")
from sklearn.preprocessing import LabelEncoder
feature_encoders = {}
X_numeric = pd.DataFrame()

for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        # Keep numeric columns as-is
        X_numeric[col] = X[col]
    else:
        # Encode non-numeric columns
        le = LabelEncoder()
        X_numeric[col] = le.fit_transform(X[col].astype(str))
        feature_encoders[col] = le
        print(f"  Encoded column '{col}': {len(le.classes_)} unique values")

# Convert to numpy array and ensure all values are numeric
X = X_numeric.values.astype(np.float64)
print(f"Features shape: {X.shape}, Features dtype: {X.dtype}")

# Extract unique class names from the dataset
class_names = sorted(y.unique().tolist())
print(f"\nFound {len(class_names)} unique classes in the dataset:")
# Only print first 20 class names to avoid excessive output
for i, class_name in enumerate(class_names[:20]):
    print(f"  {i}: {class_name}")
if len(class_names) > 20:
    print(f"  ... and {len(class_names) - 20} more classes")

# Convert text labels (e.g., 'Benign') into integers (0, 1, 2...).
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
print(f"Labels encoded. Found {len(class_names)} classes.")

# Use stratify=y_encoded to ensure class distribution is similar in train and test sets.
# Note: Stratification requires at least 2 samples per class. If many classes have only 1 sample,
# stratification will fail, so we check if it's possible first.
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print("Data split into training and testing sets (80/20 split) with stratification.")
except ValueError:
    # If stratification fails (e.g., some classes have only 1 sample), split without stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    print("Data split into training and testing sets (80/20 split) without stratification (some classes have only 1 sample).")

try:
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    print("Data split into validation and testing sets (50/50 split) with stratification.")
except ValueError:
    # If stratification fails, split without stratification
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    print("Data split into validation and testing sets (50/50 split) without stratification (some classes have only 1 sample).")



y_val = y_val.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

def count_class(partition, name):
    _, num_ins = np.unique(partition, return_counts=True)
    for value, count in zip(class_names, num_ins):
        print(f"{name}: Class {value} : {count} times")

#count_class(y_val,"val")
#count_class(y_train,"train")
#count_class(y_test,"test")

train = np.concatenate((X_train, y_train), axis=1)
test = np.concatenate((X_test, y_test), axis=1)
val = np.concatenate((X_val, y_val), axis=1)

# Final verification: Check for any remaining NaN or Inf values
print("\n--- Final Data Verification ---")
for name, array in [('train', train), ('test', test), ('val', val)]:
    # Only check for NaN/Inf if the array is numeric
    if np.issubdtype(array.dtype, np.number):
        nan_count = np.isnan(array).sum()
        inf_count = np.isinf(array).sum()
        if nan_count > 0:
            print(f"WARNING: {name} still contains {nan_count} NaN values!")
            # Fix any remaining NaN
            array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
            if name == 'train':
                train = array
            elif name == 'test':
                test = array
            else:
                val = array
            print(f"  Fixed: Replaced NaN/Inf with 0")
        elif inf_count > 0:
            print(f"WARNING: {name} still contains {inf_count} Inf values!")
            # Fix any remaining Inf
            array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
            if name == 'train':
                train = array
            elif name == 'test':
                test = array
            else:
                val = array
            print(f"  Fixed: Replaced Inf with 0")
        else:
            print(f"{name}: No NaN or Inf values ✓")
    else:
        # For non-numeric arrays, skip NaN/Inf checking
        print(f"{name}: Non-numeric array, skipping NaN/Inf check ✓")

np.save("train.npy", train)
np.save("test.npy", test)
np.save("val.npy", val)

np.save("class_names.npy", class_names) # Save class names for the final report

for name, array in [('train', train), ('test', test), ('val', val)]: print(f"{name} shape: {array.shape}")

end_time = time.time()
print("\n--- Preprocessing Complete ---")
print("Saved 3 files")
print(f"Total preprocessing time: {(end_time - start_time):.2f} seconds.")