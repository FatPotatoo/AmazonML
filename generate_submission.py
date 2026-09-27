import os
import sys
import gc
import time
import math
import numpy as np
import polars as pl
import lightgbm as lgb
from collections import defaultdict
from rapidfuzz import fuzz, distance

sys.stdout.reconfigure(encoding='utf-8')
from normalizer import normalize_text, parse_address_anchors, extract_name_tokens, extract_prefix_shingles, extract_phonetic_keys
from candidate_generator import MultiPassCandidateGenerator, extract_compound_numbers
from feature_extractor import FEATURE_NAMES

TEST_DIR = "dataset/test"
MODEL_PATH = "models/lgb_reranker_v2.txt"
OUTPUT_DIR = "output"
MATCHING_FILE = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

PRED_THRESHOLD = 0.70
SINGLETON_THRESHOLD = 0.60
MAX_MATCHES_CAP = 8
TOP_K = 25
BATCH_SIZE = 25000

def process_country(country: str, model: lgb.Booster, out_match_f, out_cand_f):
    print(f"\n{'='*65}\n  STARTING PIPELINE FOR COUNTRY: {country}\n{'='*65}", flush=True)
    t_start = time.time()
    
    # 1. Load Country Targets (S2 + S3)
    print(f"[{country}] 1. Loading target records...", flush=True)
    t0 = time.time()
    s2_df = pl.scan_csv(f"{TEST_DIR}/test_source2.tsv", separator="\t").filter(pl.col("country") == country).collect()
    s3_df = pl.scan_csv(f"{TEST_DIR}/test_source3.tsv", separator="\t").filter(pl.col("country") == country).collect()
    targets_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    gc.collect()
    n_targets = len(targets_df)
    print(f"[{country}] Loaded {n_targets:,} target records in {time.time()-t0:.2f}s.", flush=True)
    
    # 2. Build Inverted Index
    print(f"[{country}] 2. Building inverted candidate index (top_k={TOP_K})...", flush=True)
    t0 = time.time()
    cg = MultiPassCandidateGenerator(top_k=TOP_K, max_token_freq=3000, max_shingle_freq=1500, max_phonetic_freq=400)
    cg.build_target_index(targets_df)
    print(f"[{country}] Inverted index constructed in {time.time()-t0:.2f}s.", flush=True)
    
    # 3. Pre-extract Target Clean Strings & Anchors for Fast Feature Extraction
    print(f"[{country}] 3. Precomputing target clean attributes...", flush=True)
    t0 = time.time()
    tgt_ids = targets_df["entity_id"].to_list()
    tgt_names_raw = targets_df["business_name"].to_list()
    tgt_addrs_raw = targets_df["business_address"].to_list()
    
    tgt_clean_names = [normalize_text(n) for n in tgt_names_raw]
    tgt_clean_addrs = [normalize_text(a) for a in tgt_addrs_raw]
    tgt_name_tokens = [n.split() for n in tgt_clean_names]
    tgt_num_sets = []
    tgt_pcodes = []
    tgt_is_empty = []
    for a in tgt_addrs_raw:
        anchors = parse_address_anchors(a)
        tgt_num_sets.append(set(anchors['numbers']))
        tgt_pcodes.append(anchors['postal_code'])
        tgt_is_empty.append(1.0 if anchors['is_empty'] else 0.0)
    print(f"[{country}] Target features precomputed in {time.time()-t0:.2f}s.", flush=True)
    
    # Free raw targets dataframe
    del targets_df, tgt_names_raw, tgt_addrs_raw
    gc.collect()
    
    # 4. Load S1 Test Entities for Country
    print(f"[{country}] 4. Loading S1 reference entities...", flush=True)
    s1_df = pl.scan_csv(f"{TEST_DIR}/test_source1.tsv", separator="\t").filter(pl.col("country") == country).collect()
    n_s1 = len(s1_df)
    s1_ids = s1_df["entity_id"].to_list()
    s1_names = s1_df["business_name"].to_list()
    s1_addrs = s1_df["business_address"].to_list()
    del s1_df
    gc.collect()
    print(f"[{country}] Loaded {n_s1:,} S1 entities to process.", flush=True)
    
    # 5. Process S1 Entities in Batches
    print(f"[{country}] 5. Processing S1 entities in batches of {BATCH_SIZE:,}...", flush=True)
    n_batches = math.ceil(n_s1 / BATCH_SIZE)
    
    total_matched_links = 0
    total_candidate_links = 0
    
    for b_idx in range(n_batches):
        b_start = b_idx * BATCH_SIZE
        b_end = min(b_start + BATCH_SIZE, n_s1)
        b_len = b_end - b_start
        t_b0 = time.time()
        
        batch_ids = s1_ids[b_start:b_end]
        batch_names = s1_names[b_start:b_end]
        batch_addrs = s1_addrs[b_start:b_end]
        
        # Batch S1 precomputations
        batch_clean_names = [normalize_text(n) for n in batch_names]
        batch_clean_addrs = [normalize_text(a) for a in batch_addrs]
        batch_name_tokens = [n.split() for n in batch_clean_names]
        batch_num_sets = []
        batch_pcodes = []
        batch_is_empty = []
        for a in batch_addrs:
            anchors = parse_address_anchors(a)
            batch_num_sets.append(set(anchors['numbers']))
            batch_pcodes.append(anchors['postal_code'])
            batch_is_empty.append(1.0 if anchors['is_empty'] else 0.0)
            
        # Create mini-dataframe for cg query
        batch_s1_df = pl.DataFrame({
            "entity_id": batch_ids,
            "business_name": batch_names,
            "business_address": batch_addrs
        })
        
        # A. Candidate Generation
        cands_dict = cg.generate_candidates_for_s1(batch_s1_df)
        
        # B. Flatten pairs for feature extraction
        pair_s1_local_indices = []
        pair_tgt_int_indices = []
        pair_ranks = []
        pair_s1_ids = []
        pair_tgt_ids = []
        
        tgt_id_to_int = {tid: i for i, tid in enumerate(tgt_ids)}
        
        batch_candidates_list = []
        for local_i, s1_id in enumerate(batch_ids):
            c_list = cands_dict.get(s1_id, [])
            batch_candidates_list.append(c_list)
            for rank, tid in enumerate(c_list, 1):
                tgt_i = tgt_id_to_int.get(tid)
                if tgt_i is not None:
                    pair_s1_local_indices.append(local_i)
                    pair_tgt_int_indices.append(tgt_i)
                    pair_ranks.append(rank)
                    pair_s1_ids.append(s1_id)
                    pair_tgt_ids.append(tid)
                    
        n_pairs = len(pair_s1_local_indices)
        
        # C. Vectorized RapidFuzz Feature Extraction (20 features)
        if n_pairs > 0:
            X_batch = np.zeros((n_pairs, 20), dtype=np.float32)
            
            for p_i in range(n_pairs):
                s_idx = pair_s1_local_indices[p_i]
                t_idx = pair_tgt_int_indices[p_i]
                rk = pair_ranks[p_i]
                
                # Names
                sn = batch_clean_names[s_idx]
                tn = tgt_clean_names[t_idx]
                
                n_ratio = fuzz.ratio(sn, tn) / 100.0
                n_tsort = fuzz.token_sort_ratio(sn, tn) / 100.0
                n_tset = fuzz.token_set_ratio(sn, tn) / 100.0
                n_jw = distance.JaroWinkler.similarity(sn, tn)
                n_part = fuzz.partial_ratio(sn, tn) / 100.0
                n_len_diff = abs(len(sn) - len(tn))
                n_exact = 1.0 if (sn and sn == tn) else 0.0
                
                # Token alignment
                s_toks = batch_name_tokens[s_idx]
                t_toks = tgt_name_tokens[t_idx]
                first_tok = 1.0 if (s_toks and t_toks and s_toks[0] == t_toks[0]) else 0.0
                shared_toks = float(len(set(s_toks) & set(t_toks)))
                
                # Addresses
                s_empty = batch_is_empty[s_idx]
                t_empty = tgt_is_empty[t_idx]
                
                # Postal code match & mismatch penalty
                s_p = batch_pcodes[s_idx]
                t_p = tgt_pcodes[t_idx]
                if s_p and t_p:
                    if s_p == t_p:
                        p_match = 1.0
                        p_pfx = 1.0
                    else:
                        p_match = -1.0 # Hard branch mismatch penalty!
                        p_pfx = 1.0 if s_p[:3] == t_p[:3] else 0.0
                else:
                    p_match = 0.0
                    p_pfx = 0.0
                
                if s_empty == 1.0 or t_empty == 1.0:
                    a_tset = 0.0
                    a_tsort = 0.0
                    a_jw = 0.0
                    num_jaccard = 0.0
                    num_exact = 0.0
                    is_empty = 1.0
                else:
                    sa = batch_clean_addrs[s_idx]
                    ta = tgt_clean_addrs[t_idx]
                    a_tset = fuzz.token_set_ratio(sa, ta) / 100.0
                    a_tsort = fuzz.token_sort_ratio(sa, ta) / 100.0
                    a_jw = distance.JaroWinkler.similarity(sa, ta)
                    
                    s_nums = batch_num_sets[s_idx]
                    t_nums = tgt_num_sets[t_idx]
                    intersect = len(s_nums & t_nums)
                    union = len(s_nums | t_nums)
                    num_jaccard = (intersect / union) if union > 0 else 0.0
                    num_exact = 1.0 if intersect > 0 else 0.0
                    is_empty = 0.0
                    
                n_x_a = n_tset * a_tset if is_empty == 0.0 else n_tset * 0.5
                
                X_batch[p_i, 0] = n_ratio
                X_batch[p_i, 1] = n_tsort
                X_batch[p_i, 2] = n_tset
                X_batch[p_i, 3] = n_jw
                X_batch[p_i, 4] = n_part
                X_batch[p_i, 5] = n_len_diff
                X_batch[p_i, 6] = n_exact
                X_batch[p_i, 7] = first_tok
                X_batch[p_i, 8] = shared_toks
                X_batch[p_i, 9] = a_tset
                X_batch[p_i, 10] = a_tsort
                X_batch[p_i, 11] = a_jw
                X_batch[p_i, 12] = num_jaccard
                X_batch[p_i, 13] = num_exact
                X_batch[p_i, 14] = p_match
                X_batch[p_i, 15] = p_pfx
                X_batch[p_i, 16] = is_empty
                X_batch[p_i, 17] = n_x_a
                X_batch[p_i, 18] = 0.0  # block_score
                X_batch[p_i, 19] = float(rk)
                
            # D. Model Inference
            probs = model.predict(X_batch)
            
            # Group candidate scores by S1 ID
            cands_by_s1 = defaultdict(list)
            for p_i in range(n_pairs):
                cands_by_s1[pair_s1_ids[p_i]].append((pair_tgt_ids[p_i], probs[p_i]))
        else:
            cands_by_s1 = defaultdict(list)
            
        # E. Entity-Level Decision Logic (Singleton Guard + Dynamic Match Cap)
        for local_i, s1_id in enumerate(batch_ids):
            cands = batch_candidates_list[local_i]
            scored_cands = cands_by_s1.get(s1_id, [])
            
            # Singleton Guard: if highest candidate probability is low, predict singleton (empty)!
            max_prob = max((pr for _, pr in scored_cands), default=0.0)
            if max_prob < SINGLETON_THRESHOLD:
                valid_matches = []
            else:
                # Filter candidates passing threshold, sorted descending by prob, capped at MAX_MATCHES_CAP
                passing = [tid for tid, pr in sorted(scored_cands, key=lambda x: x[1], reverse=True) if pr >= PRED_THRESHOLD]
                cand_set = set(cands)
                valid_matches = [m for m in passing if m in cand_set][:MAX_MATCHES_CAP]
            
            # Candidate row
            cands_str = ",".join(cands)
            out_cand_f.write(f"{s1_id}\t{cands_str}\n")
            total_candidate_links += len(cands)
            
            # Matching row
            matches_str = ",".join(valid_matches)
            out_match_f.write(f"{s1_id}\t{matches_str}\n")
            total_matched_links += len(valid_matches)
            
        out_cand_f.flush()
        out_match_f.flush()
        
        t_b_elapsed = time.time() - t_b0
        print(f"  [{country}] Batch {b_idx+1}/{n_batches} ({b_len:,} entities, {n_pairs:,} pairs) done in {t_b_elapsed:.1f}s. "
              f"(Matches so far: {total_matched_links:,})", flush=True)
              
        # Free batch memory
        del batch_s1_df, cands_dict, pair_s1_local_indices, pair_tgt_int_indices
        if n_pairs > 0:
            del X_batch, probs
        gc.collect()
        
    print(f"\n[{country}] COMPLETED in {time.time()-t_start:.1f}s! Total Matches: {total_matched_links:,}, Candidates: {total_candidate_links:,}", flush=True)
    
    # Free country memory
    del cg, tgt_ids, tgt_clean_names, tgt_clean_addrs, tgt_num_sets, tgt_is_empty
    gc.collect()

def main():
    print("=" * 65)
    print("      AMAZON ML CHALLENGE 2026: SUBMISSION GENERATOR")
    print("=" * 65)
    t_global_start = time.time()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Model
    print(f"Loading LightGBM model from {MODEL_PATH}...", flush=True)
    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"Model loaded successfully! Classification threshold: {PRED_THRESHOLD}", flush=True)
    
    # Open submission files and write headers
    with open(MATCHING_FILE, "w", encoding="utf-8") as out_match_f, \
         open(CANDIDATE_FILE, "w", encoding="utf-8") as out_cand_f:
         
        out_match_f.write("source1_entity_id\tmatched_entity_ids\n")
        out_cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
        
        # Process Countries: France first (zero-shot test), then US, then India
        countries = ["France", "US", "India"]
        for c in countries:
            process_country(c, model, out_match_f, out_cand_f)
            
    total_time = time.time() - t_global_start
    print("\n" + "=" * 65)
    print(f"ALL COUNTRIES COMPLETED IN {total_time/60:.2f} MINUTES!")
    print(f"Output Files Generated:")
    print("=" * 65)
    
    print("\nRunning official submission validation...", flush=True)
    import subprocess
    cmd = [
        sys.executable,
        "student_resource/utils/validate_submission.py",
        "--matching", MATCHING_FILE,
        "--candidate", CANDIDATE_FILE,
        "--test-dir", "dataset/test"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

if __name__ == "__main__":
    main()
