import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("ethiopia.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# -------------------------------
# 2. Handle Missing Values (-999)
# -------------------------------
df = df.replace(-999, np.nan)

# -------------------------------
# 3. Missing Values Analysis
# -------------------------------
print("\nMissing Values (Count):")
print(df.isna().sum())

print("\nMissing Values (%):")
missing_percent = df.isna().sum() / len(df) * 100
print(missing_percent.sort_values(ascending=False))

# -------------------------------
# 4. Check Duplicates
# -------------------------------
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# -------------------------------
# 5. Summary Statistics
# -------------------------------
print("\nSummary Statistics:")
print(df.describe())

# -------------------------------
# 6. Convert YEAR + DOY to Date
# -------------------------------
df["date"] = pd.to_datetime(df["YEAR"] * 1000 + df["DOY"], format="%Y%j")

print("\nDate conversion successful:")
print(df[["YEAR", "DOY", "date"]].head())

# Set date as index (important for time series)
df.set_index("date", inplace=True)

# Monthly average temperature
monthly_temp = df["T2M"].resample("ME").mean()      
# Plot
plt.figure()
monthly_temp.plot()
plt.title("Monthly Average Temperature in Ethiopia")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.show()

monthly_rain = df["PRECTOTCORR"].resample("ME").sum()

plt.figure()
monthly_rain.plot()
plt.title("Monthly Total Rainfall in Ethiopia")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.show()