import pandas as pd

def add_header(month, hour):
    data = pd.read_csv(f"/Users/Maarten/Documents/Projects/Temperature-Model/Data/Final_data/{month+1}_{hour+1}.txt", header=None)
    data.to_csv(f"/Users/Maarten/Documents/Projects/Temperature-Model/Data/Final_data/{month+1}_{hour+1}.txt", header=["MM", "HH", "DD", "FF", "T", "TD", "RH", "N", "TC"], index=False)

[add_header(month, hour) for month in range(12) for hour in range(24)]