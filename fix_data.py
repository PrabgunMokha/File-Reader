import pandas as pd
import os

def clean_data():
    if os.path.exists("TrackData.csv"):
        # Fixed encoding to handle names like 'Anitha' correctly
        df = pd.read_csv("TrackData.csv", encoding='latin-1')
        df = df.dropna(how='all')
        df.to_csv("TrackData_Clean.csv", index=False)
        print("✅ TrackData_Clean.csv created successfully!")
    else:
        print("❌ Error: Put 'TrackData.csv' in this folder first.")

if __name__ == "__main__":
    clean_data()