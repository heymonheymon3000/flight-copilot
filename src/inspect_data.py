import pandas as pd
from src.paths import DATA_RAW

df = pd.read_csv(DATA_RAW / "asrs_air_carrier_2025_2026.csv", header=[0, 1])
print("Columns:")
for col in df.columns:
    print(col)

print("\nRow count:", len(df))
print("\nFirst row, Report columns only:")
for col in df.columns:
    if col[0].startswith("Report"):
        print(f"\n--- {col} ---")
        print(df.iloc[0][col])
