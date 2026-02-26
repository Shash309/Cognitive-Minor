import pandas as pd
import glob
import os

files = glob.glob("datasets/*.csv")  # all domain CSVs

dfs = []
for file in files:
    field = os.path.splitext(os.path.basename(file))[0].lower()
    df = pd.read_csv(file)

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # map Kaggle columns to standard names
    rename_map = {
        "name": "college_name",
        "state": "state",
        "city": "city",
        "rank": "ranking",
        "score": "score"
    }
    df = df.rename(columns=rename_map)

    # check required column
    if "college_name" not in df.columns:
        print(f"⚠️ Skipping {file}, no college name column found")
        continue

    # add field column
    df["field"] = field

    # select only the useful columns (extra optional ones kept)
    keep_cols = ["college_name", "state", "city", "field", "ranking", "score"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None

    dfs.append(df[keep_cols])

# merge all
if dfs:
    final = pd.concat(dfs, ignore_index=True)
    final.to_csv("colleges.csv", index=False)
    print("✅ Merged dataset saved as colleges.csv with", len(final), "rows")
else:
    print("⚠️ No valid CSVs found to merge")
