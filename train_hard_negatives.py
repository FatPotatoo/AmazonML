import os
import sys
import gc
import time
import math
import numpy as np
import polars as pl
import lightgbm as lgb
from collections import defaultdict
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')
from candidate_generator import MultiPassCandidateGenerator
from feature_extractor import extract_features_for_pairs, FEATURE_NAMES

def calculate_macro_f05(gt_map: dict[str, list[str]], pred_map: dict[str, list[str]]) -> float:
    scores = []
    for s1_id, true_matches in gt_map.items():
        true_set = set(true_matches)
        pred_set = set(pred_map.get(s1_id, []))
        
        if not true_set:
            scores.append(1.0 if not pred_set else 0.0)
            continue
            
        if not pred_set:
            scores.append(0.0)
            continue
            
        tp = len(true_set & pred_set)
        if tp == 0:
            scores.append(0.0)
            continue
            
        p = tp / len(pred_set)
        r = tp / len(true_set)
        denom = 0.25 * p + r
        scores.append((1.25 * p * r / denom) if denom > 0 else 0.0)
        
    return float(np.mean(scores))

def main():
    print("=" * 65)
    print("   LIGHTGBM TRAINING WITH REAL FULL-SCALE HARD NEGATIVES")
    print("=" * 65)
    t_global_start = time.time()
    
    # 1. Sample S1 entities from train (8,000 India, 8,000 US = 16,000 total)
    print("1. Sampling 16,000 S1 training entities (8,000 India, 8,000 US)...", flush=True)
    s1_all = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
    s1_in = s1_all.filter(pl.col("country") == "India").head(8000)
    s1_us = s1_all.filter(pl.col("country") == "US").head(8000)
    del s1_all
    gc.collect()
    
    # Load Ground Truth
    print("2. Loading ground truth matching labels...", flush=True)
    gt_all = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
    sampled_s1_ids = set(s1_in["entity_id"].to_list() + s1_us["entity_id"].to_list())
    gt_filtered = gt_all.filter(pl.col("source1_entity_id").is_in(list(sampled_s1_ids)))
    del gt_all
    gc.collect()
    
    gt_map = {}
    for r in gt_filtered.iter_rows(named=True):
        m = r["matched_entity_ids"]
        if m and str(m).strip() and str(m) != "None":
            gt_map[r["source1_entity_id"]] = [x.strip() for x in str(m).split(",") if x.strip()]
        else:
            gt_map[r["source1_entity_id"]] = []
            
    n_singletons = sum(1 for v in gt_map.values() if len(v) == 0)
    total_true_pairs = sum(len(v) for v in gt_map.values())
    print(f"Sampled 16,000 S1: {total_true_pairs:,} true links, {n_singletons:,} singletons ({n_singletons/16000*100:.1f}%).", flush=True)
    
    # 3. Mine Candidates & Hard Negatives from Full Target Space
    all_pairs = []
    all_s1_dict = {}
    all_tgt_dict = {}
    
    for country, s1_sample in [("India", s1_in), ("US", s1_us)]:
        print(f"\n--- Mining Hard Negatives for {country} ---", flush=True)
        t0 = time.time()
        
        # Load targets for this country
        s2 = pl.scan_csv("dataset/train/train_source2.tsv", separator="\t").filter(pl.col("country") == country).collect()
        s3 = pl.scan_csv("dataset/train/train_source3.tsv", separator="\t").filter(pl.col("country") == country).collect()
        targets_c = pl.concat([s2, s3])
        del s2, s3
        gc.collect()
        print(f"Loaded {len(targets_c):,} full {country} targets in {time.time()-t0:.1f}s.", flush=True)
        
        # Build candidate generator
        cg = MultiPassCandidateGenerator(top_k=25, max_token_freq=3000, max_shingle_freq=1500, max_phonetic_freq=400)
        t_b0 = time.time()
        cg.build_target_index(targets_c)
        print(f"Index built in {time.time()-t_b0:.1f}s. Querying {len(s1_sample)} S1 entities...", flush=True)
        
        t_q0 = time.time()
        cands_dict = cg.generate_candidates_for_s1(s1_sample)
        print(f"Querying completed in {time.time()-t_q0:.1f}s.", flush=True)
        
        # Collect only targets that were actually retrieved to save memory
        retrieved_tids = set()
        for cand_list in cands_dict.values():
            retrieved_tids.update(cand_list)
            
        print(f"Collecting features for {len(retrieved_tids):,} unique retrieved targets...", flush=True)
        retrieved_targets = targets_c.filter(pl.col("entity_id").is_in(list(retrieved_tids)))
        for r in retrieved_targets.iter_rows(named=True):
            all_tgt_dict[r["entity_id"]] = r
        del retrieved_targets, retrieved_tids
        
        for r in s1_sample.iter_rows(named=True):
            all_s1_dict[r["entity_id"]] = r
            
        # Build candidate pairs with True Hard Negative labels
        c_recalled = 0
        c_total_true = 0
        country_s1_ids = s1_sample["entity_id"].to_list()
        
        for sid in country_s1_ids:
            true_set = set(gt_map.get(sid, []))
            c_total_true += len(true_set)
            cand_list = cands_dict.get(sid, [])
            for rk, tid in enumerate(cand_list, 1):
                is_pos = 1 if tid in true_set else 0
                if is_pos:
                    c_recalled += 1
                all_pairs.append({
                    "s1_id": sid,
                    "tgt_id": tid,
                    "rank": rk,
                    "label": is_pos
                })
                
        del cg, targets_c
        gc.collect()
        print(f"{country} Recall Ceiling: {c_recalled/c_total_true*100:.2f}% ({c_recalled:,}/{c_total_true:,})", flush=True)
        
    print(f"\nTotal candidate pairs mined: {len(all_pairs):,} (Positives: {sum(p['label'] for p in all_pairs):,})")
    
    # 4. Group-aware Train / Eval Split by S1 ID
    print("\n4. Splitting 80% Train / 20% Eval (group-aware by S1 ID)...", flush=True)
    np.random.seed(42)
    s1_id_list = list(sampled_s1_ids)
    np.random.shuffle(s1_id_list)
    split_idx = int(0.80 * len(s1_id_list))
    train_s1_ids = set(s1_id_list[:split_idx])
    eval_s1_ids = set(s1_id_list[split_idx:])
    
    train_pairs = [p for p in all_pairs if p["s1_id"] in train_s1_ids]
    eval_pairs = [p for p in all_pairs if p["s1_id"] in eval_s1_ids]
    print(f"  Train: {len(train_pairs):,} pairs ({len(train_s1_ids):,} entities)")
    print(f"  Eval:  {len(eval_pairs):,} pairs ({len(eval_s1_ids):,} entities)")
    
    # 5. Extract Features
    print("\n5. Extracting 20 features...", flush=True)
    t_f0 = time.time()
    X_train, y_train = extract_features_for_pairs(train_pairs, all_s1_dict, all_tgt_dict)
    X_eval, y_eval = extract_features_for_pairs(eval_pairs, all_s1_dict, all_tgt_dict)
    print(f"Features extracted in {time.time()-t_f0:.1f}s. Matrix shape: {X_train.shape}")
    
    del all_s1_dict, all_tgt_dict
    gc.collect()
    
    # 6. Train LightGBM
    print("\n6. Training LightGBM Classifier...", flush=True)
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    eval_data = lgb.Dataset(X_eval, label=y_eval, reference=train_data, feature_name=FEATURE_NAMES)
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=400,
        valid_sets=[train_data, eval_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=50)]
    )
    
    os.makedirs("models", exist_ok=True)
    model.save_model("models/lgb_reranker_v2.txt")
    print("\nSaved new model to models/lgb_reranker_v2.txt")
    
    # Feature importances
    print("\n--- FEATURE IMPORTANCES ---")
    importances = model.feature_importance(importance_type="gain")
    for feat, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<22}: {imp:,.1f}")
        
    # 7. Joint Calibration: Threshold + Singleton Guard + Dynamic Match Cap
    print("\n7. Evaluating & Calibrating on 3,200 Holdout Entities...", flush=True)
    pred_probs = model.predict(X_eval)
    auc_score = roc_auc_score(y_eval, pred_probs)
    print(f"Holdout ROC-AUC: {auc_score:.4f}")
    
    eval_gt_map = {sid: gt_map[sid] for sid in eval_s1_ids}
    
    # Group candidates by S1 ID
    eval_by_s1 = defaultdict(list)
    for i, p in enumerate(eval_pairs):
        eval_by_s1[p["s1_id"]].append((p["tgt_id"], pred_probs[i]))
        
    best_f05 = 0.0
    best_config = None
    
    print("\n" + "=" * 80)
    print(f"{'Match Thresh':<14} | {'Singleton Thresh':<18} | {'Max Cap':<9} | {'Macro F0.5':<11} | {'Status'}")
    print("-" * 80)
    
    for m_th in [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
        for s_th in [0.60, 0.70, 0.75, 0.80, 0.85]:
            for cap in [4, 5, 6, 8]:
                pred_map = {}
                for sid in eval_s1_ids:
                    cands = eval_by_s1.get(sid, [])
                    if not cands:
                        pred_map[sid] = []
                        continue
                        
                    # Singleton guard
                    max_prob = max(prob for _, prob in cands)
                    if max_prob < s_th:
                        pred_map[sid] = []
                        continue
                        
                    # Filter matches above threshold, sorted by prob descending, capped
                    valid = [tid for tid, prob in sorted(cands, key=lambda x: x[1], reverse=True) if prob >= m_th][:cap]
                    pred_map[sid] = valid
                    
                score = calculate_macro_f05(eval_gt_map, pred_map)
                if score > best_f05:
                    best_f05 = score
                    best_config = (m_th, s_th, cap)
                    print(f"{m_th:<14.2f} | {s_th:<18.2f} | {cap:<9d} | {score:<11.4f} | *** NEW BEST ***")
                    
    print("=" * 80)
    print(f"Optimal Match Threshold:     {best_config[0]:.2f}")
    print(f"Optimal Singleton Threshold: {best_config[1]:.2f}")
    print(f"Optimal Match Cap:           {best_config[2]}")
    print(f"Calibrated Macro F0.5 Score: {best_f05:.4f}")
    print(f"Pipeline finished in {(time.time()-t_global_start)/60:.1f} minutes.")
    print("=" * 80)

if __name__ == "__main__":
    main()
