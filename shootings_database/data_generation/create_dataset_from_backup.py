import glob
import os
import pandas as pd

file_dir = r".\data\backup"
file_str = 'Michigan_Lansing_COMPLAINTS'
output_dir = r"..\opd-datasets\data"

files = glob.glob(os.path.join(file_dir, file_str+"*"))
df = []
for f in files:
    new_df = pd.read_csv(f)
    df.append(new_df)

df = pd.concat(df)

output_file = os.path.join(output_dir, file_str+".csv")
df.to_csv(output_file, index=False)