from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import ast
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import re

# load dataset and combine splits
dataset = load_dataset("ai4privacy/pii-masking-300k")  
dataframes = []
for split_name in dataset.keys():
    df = dataset[split_name].to_pandas() 
    # df['split'] = split_name  # optional keep track of which split each row came from
    dataframes.append(df)

merged_df = pd.concat(dataframes, ignore_index=True)
# all languages no split
merged_df.to_csv("merged_dataset.csv", index=False)
print("Dataset saved to 'merged_dataset.csv'")

# remove set column 
merged_df = merged_df.drop('set', axis=1)

# print columns for above
print("\nColumns in the CSV file:")
print(merged_df.columns.tolist())

# filter rows where 'language' == ('English' or 'Spanish')
if 'language' in merged_df.columns:
    filtered_df = merged_df[merged_df['language'].isin(['English', 'Spanish'])]   
    print(f"\nFound {len(filtered_df)} rows where 'language' == 'english' or 'spanish")
    filtered_df.to_csv("filtered_english_and_spanish_rows.csv", index=False)
    print("Filtered data saved to 'filtered_english_and_spanish_rows.csv'")
else:
    print("\nError: Column 'language' not found in the dataset.")
    print("Available columns:", merged_df.columns.tolist())   


def parse_span_labels(x):
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return x

def extract_types(span_list):
    return [span[2] for span in span_list if len(span) >= 3]

def extract_group_id(row_id):
    match = re.match(r"(\d+)", str(row_id))
    return match.group(1) if match else str(row_id)

filtered_df = filtered_df.copy()

# Keep a standard text column for downstream scripts.
if "text" not in filtered_df.columns and "source_text" in filtered_df.columns:
    filtered_df["text"] = filtered_df["source_text"]

filtered_df["span_labels_parsed"] = filtered_df["span_labels"].apply(parse_span_labels)
filtered_df["label_types"] = filtered_df["span_labels_parsed"].apply(extract_types)

# Drop CARDISSUER rows (<25 instances so too little data to analyze)
filtered_df = filtered_df[
    ~filtered_df["label_types"].apply(lambda lst: "CARDISSUER" in lst)
].reset_index(drop=True)

# Normalize language
filtered_df["language_norm"] = (
    filtered_df["language"].astype(str).str.lower().str.strip()
)

# Extract group_id from row_id
filtered_df["group_id"] = filtered_df["id"].apply(extract_group_id)

# group level labels
grouped = filtered_df.groupby("group_id")

# Merge labels within group
group_labels = grouped["label_types"].apply(
    lambda lists: list(set(l for sub in lists for l in sub))
)

# Take first language
group_language = grouped["language_norm"].first()

# Multilabel binarizer
mlb = MultiLabelBinarizer(sparse_output=False)
Y_labels_group = mlb.fit_transform(group_labels)

# Language column (english=1, spanish=0)
Y_format_group = (group_language == "english").astype(int).to_numpy().reshape(-1, 1)

# Combined stratification target
Y_combined_group = np.hstack([Y_labels_group, Y_format_group])

group_ids = group_labels.index.to_numpy()

# Print overall stats
label_classes = list(mlb.classes_)
label_counts = Y_labels_group.sum(axis=0).astype(int)

print("Number of unique labels found:", len(label_classes))
print("Labels:", label_classes)
print("Counts per label:")
for label, count in zip(label_classes, label_counts):
    print(f"  {label}: {count}")

print("Overall data_format counts:")
print("  English:", int(Y_format_group.sum()))
print("  Spanish:", int(len(Y_format_group) - Y_format_group.sum()))

# splitting on group 100 into 70 and 30
msss = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=0.30, random_state=42
)

train_g_idx, temp_g_idx = next(
    msss.split(group_ids, Y_combined_group)
)

train_groups = group_ids[train_g_idx]
temp_groups = group_ids[temp_g_idx]

# splitting on group 30 into 15 and 15
Y_combined_temp = Y_combined_group[temp_g_idx]

msss2 = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=0.5, random_state=42
)

val_g_idx, test_g_idx = next(
    msss2.split(temp_groups, Y_combined_temp)
)

val_groups = temp_groups[val_g_idx]
test_groups = temp_groups[test_g_idx]

# map back to rows
df_train = filtered_df[
    filtered_df["group_id"].isin(train_groups)
].reset_index(drop=True)

df_val = filtered_df[
    filtered_df["group_id"].isin(val_groups)
].reset_index(drop=True)

df_test = filtered_df[
    filtered_df["group_id"].isin(test_groups)
].reset_index(drop=True)

def validate_and_drop_invalid_ids(df, id_column='id'):
    
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in CSV.")

    # Split ID into numeric and letter parts
    def split_id(id_val):
        match = re.fullmatch(r'(\d+)([A-Z]*)', str(id_val).strip())
        if match:
            return match.group(1), match.group(2)
        return str(id_val), ''

    df[['numeric', 'letter']] = df[id_column].apply(lambda x: pd.Series(split_id(x)))

    def is_valid_group(group):
        letters = sorted([l for l in group['letter'] if l])
        if not letters:
            return True  # No letters → valid
        expected = [chr(ord('A') + i) for i in range(len(letters))]
        return letters == expected

    # Create mask to keep only valid groups
    valid_groups = df.groupby('numeric').filter(is_valid_group)

    print(f"Dropped {len(df) - len(valid_groups)} rows from invalid ID groups.")
    return valid_groups.drop(columns=['numeric', 'letter'])  

df_train = validate_and_drop_invalid_ids(df_train, 'id')
df_val = validate_and_drop_invalid_ids(df_val, 'id')
df_test = validate_and_drop_invalid_ids(df_test, 'id')

# print stats
def print_split_stats(name, df_split):
    parsed = df_split["span_labels"].apply(parse_span_labels)
    types = parsed.apply(lambda sl: [s[2] for s in sl if len(s) >= 3])

    Y_l = MultiLabelBinarizer(classes=label_classes).fit_transform(types)

    fmt = df_split["language"].astype(str).str.lower().str.strip()
    fmt_counts_english = int((fmt == "english").sum())

    print(f"{name} size: {len(df_split)}")
    print(f"  english: {fmt_counts_english}")
    print(f"  spanish: {len(df_split) - fmt_counts_english}")
    print("  label counts:")
    for label, count in zip(label_classes, Y_l.sum(axis=0).astype(int)):
        print(f"    {label}: {count}")

print_split_stats("TRAIN", df_train)
print_split_stats("VALIDATION", df_val)
print_split_stats("TEST", df_test)

df_train.to_csv("group_training.csv", index=False)
df_val.to_csv("group_validation.csv", index=False)
df_test.to_csv("group_testing.csv", index=False)
