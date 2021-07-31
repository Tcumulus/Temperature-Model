import pandas as pd

data = pd.read_csv("/Users/Maarten/Documents/Projects/Temperature-Model/Data/test.txt", delimiter = ",") # Import data

data.columns = data.columns.str.strip() # Strip spaces
data.drop(["# STN", "FH", "FX", "T10N", "SQ", "Q", "DR", "P", "VV", "U", "WW", "IX", "M", "R", "S", "O", "Y"], axis=1, inplace=True) # Drop useless parameters
data["YYYYMMDD"] = (data.YYYYMMDD.astype(str).str)[4:] # Deletes YYYY

data["NT"] = data["T"]
data["NT"] = data.NT.shift(periods=-1) # Move NT one up
data["TC"] = data["NT"] - data["T"] # Calculate hourly temperature difference # TC = T next hour - T now
data.drop(["NT"], axis=1, inplace=True)

# DATA FORMAT: MMDD HH FF T TD RH N TC
data.to_csv("/Users/Maarten/Documents/Projects/Temperature-Model/Data/temporary.txt", index=False)