import pandas as pd

def format_data(n):
    data = pd.read_csv(f"/Users/Maarten/Documents/Projects/Temperature-Model/Data/Raw_data/{n}.txt", delimiter = ",", low_memory=False) # Import data
    print(f"Opening file {n}")
    data.columns = data.columns.str.strip() # Strip spaces
    data.drop(["# STN", "FH", "FX", "T10N", "SQ", "Q", "DR", "P", "VV", "U", "WW", "IX", "M", "R", "S", "O", "Y"], axis=1, inplace=True) # Drop useless parameters
    data["YYYYMMDD"] = (data.YYYYMMDD.astype(str).str)[4:] # Deletes YYYY

    data["NT"] = data["T"]
    data["NT"] = data.NT.shift(periods=-1) # Move NT one up
    data["TC"] = data["NT"] - data["T"] # Calculate hourly temperature difference # TC = T next hour - T now
    data.drop(["NT"], axis=1, inplace=True)
    data.drop(data.tail(1).index,inplace=True) # drop last row

    # DATA FORMAT: MMDD HH FF T TD RH N TC
    if n != 0: data.to_csv(f"/Users/Maarten/Documents/Projects/Temperature-Model/Data/Filtered_data/{n}.txt", header=False, index=False)
    else: data.to_csv(f"/Users/Maarten/Documents/Projects/Temperature-Model/Data/Filtered_data/{n}.txt", header=True, index=False)

[format_data(n) for n in range(12+1)]