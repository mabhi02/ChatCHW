import pandas as pd

# Load the CSV exactly like the API does
df = pd.read_csv('data.csv')

print("CSV Row to Case ID Mapping:")
print("=" * 50)
print(f"Total rows in CSV: {len(df) + 1} (including header)")
print(f"Total data rows: {len(df)}")
print()

# Show first 25 rows to see the mapping
for i in range(min(25, len(df))):
    row = df.iloc[i]
    print(f"Case ID {i:2d}: Row {i+2:2d} in CSV -> {row['age (years)']} {row['Sex']} - {row['Complaint']} ({row['Duration']})")

print()
print("Checking specific cases you mentioned:")
print(f"Case ID 18: {df.iloc[18]['age (years)']} {df.iloc[18]['Sex']} - {df.iloc[18]['Complaint']} ({df.iloc[18]['Duration']})")
print(f"Case ID 19: {df.iloc[19]['age (years)']} {df.iloc[19]['Sex']} - {df.iloc[19]['Complaint']} ({df.iloc[19]['Duration']})")
print(f"Case ID 20: {df.iloc[20]['age (years)']} {df.iloc[20]['Sex']} - {df.iloc[20]['Complaint']} ({df.iloc[20]['Duration']})") 