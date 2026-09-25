import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import polars as pl

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
VAL_DIR = "dataset_val"
os.makedirs(VAL_DIR, exist_ok=True)

print("1. Loading ground truth sample...", flush=True)
# Sample 20,000 S1 entities (10k US, 10k India)
s1_scan = pl.scan_csv(f"{DATA_DIR}/train_source1.tsv", separator="\t")
s1_us = s1_scan.filter(pl.col("country") == "US").limit(12000).collect()
s1_in = s1_scan.filter(pl.col("country") == "India").limit(12000).collect()
s1_val = pl.concat([s1_us, s1_in])
val_s1_ids = set(s1_val["entity_id"].to_list())
print(f"Sampled {len(val_s1_ids)} S1 validation entities ({len(s1_us)} US, {len(s1_in)} India).", flush=True)

# Save validation S1
s1_val.write_csv(f"{VAL_DIR}/val_source1.tsv", separator="\t")

print("2. Filtering ground truth for validation S1...", flush=True)
gt_scan = pl.scan_csv(f"{DATA_DIR}/train_ground_truth.tsv", separator="\t")
gt_val = gt_scan.filter(pl.col("source1_entity_id").is_in(list(val_s1_ids))).collect()
gt_val.write_csv(f"{VAL_DIR}/val_ground_truth.tsv", separator="\t")

# Collect all true target IDs
true_target_ids = set()
for match_str in gt_val["matched_entity_ids"].drop_nulls():
    for mid in str(match_str).split(","):
        mid = mid.strip()
        if mid:
            true_target_ids.add(mid)
print(f"Found {len(true_target_ids)} true target IDs in ground truth.", flush=True)

print("3. Filtering S2 and S3 (including true targets + distractor pool)...", flush=True)
# Collect all true targets + 100k random distractors from S2
s2_scan = pl.scan_csv(f"{DATA_DIR}/train_source2.tsv", separator="\t")
s2_true = s2_scan.filter(pl.col("entity_id").is_in(list(true_target_ids))).collect()
s2_distractors = s2_scan.limit(100000).collect()
s2_val = pl.concat([s2_true, s2_distractors]).unique(subset=["entity_id"])
s2_val.write_csv(f"{VAL_DIR}/val_source2.tsv", separator="\t")
print(f"Validation S2 size: {len(s2_val)} records (contains all {len(s2_true)} true targets).", flush=True)

# Collect all true targets + 100k random distractors from S3
s3_scan = pl.scan_csv(f"{DATA_DIR}/train_source3.tsv", separator="\t")
s3_true = s3_scan.filter(pl.col("entity_id").is_in(list(true_target_ids))).collect()
s3_distractors = s3_scan.limit(100000).collect()
s3_val = pl.concat([s3_true, s3_distractors]).unique(subset=["entity_id"])
s3_val.write_csv(f"{VAL_DIR}/val_source3.tsv", separator="\t")
print(f"Validation S3 size: {len(s3_val)} records (contains all {len(s3_true)} true targets).", flush=True)

print("\nValidation benchmark successfully created in 'dataset_val/'!")
