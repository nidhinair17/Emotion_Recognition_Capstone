import pandas as pd
import os

csv_df = pd.read_csv("/Users/smriti/Desktop/CapProj/New/Mainfeaturedetection/Speech_Test_preprocessed_new.csv")
excel_df = pd.read_excel("/Users/smriti/Desktop/CapProj/New/Mainfeaturedetection/FaceTest_new.xlsx")

def extract_base_name(path, suffix):
    base = os.path.basename(path)  
    base = base.replace(suffix, "") 
    base = os.path.splitext(base)[0] 
    return base

csv_df["base_name"] = csv_df["Path"].apply(lambda x: extract_base_name(x, "-stretched"))
excel_df["base_name"] = excel_df["video_path"].apply(lambda x: extract_base_name(x, "-rotation"))

excel_df = excel_df.reset_index(drop=True)
excel_df["used"] = False

# Collect matched rows
matched_rows = []

for name in csv_df["base_name"]:
    # Find the first unused Excel row with matching base name
    match_idx = excel_df[(excel_df["base_name"] == name) & (~excel_df["used"])].index

    if not match_idx.empty:
        idx = match_idx[0]
        matched_rows.append(excel_df.loc[idx].drop(labels=["used", "base_name"]))
        excel_df.at[idx, "used"] = True  # mark as used
    else:
        print(f"⚠️ No unused match found in Excel for: {name}")
        # Optional: append a row of NaNs or skip — here we skip unmatched

# Create final DataFrame
reordered_df = pd.DataFrame(matched_rows)

# Save to Excel
reordered_df.to_excel("/Users/smriti/Desktop/face_features_reordered_no_extra_rows.xlsx", index=False)

print("✅ Saved as 'face_features_reordered_no_extra_rows.xlsx'")
