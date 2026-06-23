import pandas as pd

# Load the jarvis parquet file directly from your new directory structure
file_path = "data/raw/jarvis_dft3d.parquet"
df = pd.read_parquet(file_path)

# Display a quick structural overview of the dataset
print("--- Data Snapshot ---")
print(df.head())

print("\n--- Columns and Types ---")
print(df.info())