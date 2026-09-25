import os
import sys
import math
import time
import re
from itertools import combinations
from collections import defaultdict, Counter
from array import array

sys.stdout.reconfigure(encoding='utf-8')
import polars as pl
from normalizer import (
    normalize_text, 
    extract_name_tokens, 
    extract_prefix_shingles, 
    extract_phonetic_keys,
    parse_address_anchors
)

RE_COMPOUND_NUM = re.compile(r'\b[A-Za-z0-9]+[-/][A-Za-z0-9/-]+\b')

def extract_compound_numbers(text: str) -> list[str]:
    """Extract compound plot/door numbers like 39-17-48/1, J-3/299, AF-684."""
    if not text or str(text) == 'nan':
        return []
    matches = RE_COMPOUND_NUM.findall(str(text))
    return [m.lower().strip() for m in matches if any(c.isdigit() for c in m)]

class MultiPassCandidateGenerator:
    def __init__(self, top_k: int = 50, max_token_freq: int = 4000, max_shingle_freq: int = 2000, max_phonetic_freq: int = 500):
        self.top_k = top_k
        self.max_token_freq = max_token_freq
        self.max_shingle_freq = max_shingle_freq
        self.max_phonetic_freq = max_phonetic_freq

    def build_target_index(self, target_df: pl.DataFrame):
        """Build multi-channel inverted index over target records (S2 + S3)."""
        print(f"  Indexing {len(target_df)} target records...", flush=True)
        
        self.target_ids = target_df["entity_id"].to_list()
        target_names = target_df["business_name"].to_list()
        target_addrs = target_df["business_address"].to_list()
        
        self.idx_name_tokens = defaultdict(lambda: array('I'))
        self.idx_name_shingles = defaultdict(lambda: array('I'))
        self.idx_phonetics = defaultdict(lambda: array('I'))
        self.idx_addr_anchors = defaultdict(lambda: array('I'))
        
        name_token_counts = Counter()
        shingle_counts = Counter()
        phonetic_counts = Counter()
        addr_key_counts = Counter()
        
        print("  Extracting keys from targets...", flush=True)
        for i in range(len(self.target_ids)):
            raw_name = target_names[i]
            raw_addr = target_addrs[i]
            
            clean_name = normalize_text(raw_name)
            tokens = extract_name_tokens(clean_name)
            shingles = extract_prefix_shingles(tokens, min_len=3, max_len=4)
            phonetic_keys = extract_phonetic_keys(tokens)
            
            no_space_name = clean_name.replace(" ", "")
            if len(no_space_name) >= 4:
                shingles.append(no_space_name[:4])
                if len(no_space_name) >= 5:
                    shingles.append(no_space_name[:5])
            
            for tok in set(tokens):
                self.idx_name_tokens[tok].append(i)
                name_token_counts[tok] += 1
                
            # Unordered word pairs (transposition invariant)
            if 2 <= len(tokens) <= 4:
                for t1, t2 in combinations(sorted(set(tokens)), 2):
                    wp = f"wp:{t1}_{t2}"
                    self.idx_name_tokens[wp].append(i)
                    name_token_counts[wp] += 1
                
            for sh in set(shingles):
                self.idx_name_shingles[sh].append(i)
                shingle_counts[sh] += 1
                
            for pk in set(phonetic_keys):
                self.idx_phonetics[pk].append(i)
                phonetic_counts[pk] += 1
                
            # Address keys
            addr_info = parse_address_anchors(raw_addr)
            numbers = addr_info['numbers']
            pcode = addr_info['postal_code']
            addr_tokens = addr_info['clean_address_tokens']
            compounds = extract_compound_numbers(raw_addr)
            
            keys = []
            # Compound door/plot number anchor (e.g. 39-17-48/1)
            for c_num in compounds[:2]:
                keys.append(f"cpd:{c_num}")
                
            if numbers and pcode:
                for num in numbers[:3]:
                    keys.append(f"npc:{num}_{pcode}")
                
            if numbers and addr_tokens:
                for num in numbers[:3]:
                    for at in addr_tokens[:4]:
                        keys.append(f"nat:{num}_{at}")
                    
            if pcode and tokens:
                keys.append(f"pcn:{pcode}_{tokens[0][:3]}")
                
            if len(addr_tokens) >= 2:
                for a1, a2 in combinations(addr_tokens[:3], 2):
                    pair = sorted([a1, a2])
                    keys.append(f"at2:{pair[0]}_{pair[1]}")
                    
            if numbers and phonetic_keys:
                for num in numbers[:2]:
                    for pk in phonetic_keys[:2]:
                        keys.append(f"pn:{pk}_{num}")
                
            for k in set(keys):
                self.idx_addr_anchors[k].append(i)
                addr_key_counts[k] += 1

        # Prune high-frequency keys
        for tok, count in name_token_counts.items():
            if count > self.max_token_freq:
                del self.idx_name_tokens[tok]
                
        for sh, count in shingle_counts.items():
            if count > self.max_shingle_freq:
                del self.idx_name_shingles[sh]

        for pk, count in phonetic_counts.items():
            if count > self.max_phonetic_freq:
                del self.idx_phonetics[pk]

        for k, count in addr_key_counts.items():
            if count > 2000:
                del self.idx_addr_anchors[k]

        print(f"  Indexed {len(self.idx_name_tokens)} name tokens, {len(self.idx_name_shingles)} shingles, {len(self.idx_phonetics)} phonetics, {len(self.idx_addr_anchors)} address anchors.", flush=True)

    def generate_candidates_for_s1(self, s1_df: pl.DataFrame) -> dict[str, list[str]]:
        """Generate candidates for each S1 entity using the multi-pass index."""
        print(f"  Querying for {len(s1_df)} S1 entities...", flush=True)
        s1_ids = s1_df["entity_id"].to_list()
        s1_names = s1_df["business_name"].to_list()
        s1_addrs = s1_df["business_address"].to_list()
        
        candidates_out = {}
        
        for i in range(len(s1_ids)):
            s1_id = s1_ids[i]
            raw_name = s1_names[i]
            raw_addr = s1_addrs[i]
            
            clean_name = normalize_text(raw_name)
            tokens = extract_name_tokens(clean_name)
            shingles = extract_prefix_shingles(tokens, min_len=3, max_len=4)
            phonetic_keys = extract_phonetic_keys(tokens)
            
            no_space_name = clean_name.replace(" ", "")
            if len(no_space_name) >= 4:
                shingles.append(no_space_name[:4])
                if len(no_space_name) >= 5:
                    shingles.append(no_space_name[:5])
            
            addr_info = parse_address_anchors(raw_addr)
            numbers = addr_info['numbers']
            pcode = addr_info['postal_code']
            addr_tokens = addr_info['clean_address_tokens']
            compounds = extract_compound_numbers(raw_addr)
            
            name_scores = defaultdict(float)
            addr_scores = defaultdict(float)
            
            # Pass 1: Name Tokens
            for tok in set(tokens):
                if tok in self.idx_name_tokens:
                    posting = self.idx_name_tokens[tok]
                    w = 3.5 / math.log2(2 + len(posting))
                    for tgt_idx in posting:
                        name_scores[tgt_idx] += w
                        
            # Pass 1b: Unordered Word Pairs
            if 2 <= len(tokens) <= 4:
                for t1, t2 in combinations(sorted(set(tokens)), 2):
                    wp = f"wp:{t1}_{t2}"
                    if wp in self.idx_name_tokens:
                        posting = self.idx_name_tokens[wp]
                        w = 5.0 / math.log2(2 + len(posting))
                        for tgt_idx in posting:
                            name_scores[tgt_idx] += w
                        
            # Pass 2: Prefix Shingles
            for sh in set(shingles):
                if sh in self.idx_name_shingles:
                    posting = self.idx_name_shingles[sh]
                    w = 1.0 / math.log2(2 + len(posting))
                    for tgt_idx in posting:
                        name_scores[tgt_idx] += w

            # Pass 3: Phonetic Keys
            for pk in set(phonetic_keys):
                if pk in self.idx_phonetics:
                    posting = self.idx_phonetics[pk]
                    w = 1.5 / math.log2(2 + len(posting))
                    for tgt_idx in posting:
                        name_scores[tgt_idx] += w
                        
            # Pass 4: Address Anchors
            for c_num in compounds[:2]:
                k = f"cpd:{c_num}"
                if k in self.idx_addr_anchors:
                    for tgt_idx in self.idx_addr_anchors[k]:
                        addr_scores[tgt_idx] += 6.0  # Ultra high weight for compound numbers
                        
            if numbers and pcode:
                for num in numbers[:3]:
                    k = f"npc:{num}_{pcode}"
                    if k in self.idx_addr_anchors:
                        for tgt_idx in self.idx_addr_anchors[k]:
                            addr_scores[tgt_idx] += 4.5
                        
            if numbers and addr_tokens:
                for num in numbers[:3]:
                    for at in addr_tokens[:4]:
                        k = f"nat:{num}_{at}"
                        if k in self.idx_addr_anchors:
                            for tgt_idx in self.idx_addr_anchors[k]:
                                addr_scores[tgt_idx] += 4.0
                            
            if pcode and tokens:
                k = f"pcn:{pcode}_{tokens[0][:3]}"
                if k in self.idx_addr_anchors:
                    for tgt_idx in self.idx_addr_anchors[k]:
                        addr_scores[tgt_idx] += 3.5
                        
            if len(addr_tokens) >= 2:
                for a1, a2 in combinations(addr_tokens[:3], 2):
                    pair = sorted([a1, a2])
                    k = f"at2:{pair[0]}_{pair[1]}"
                    if k in self.idx_addr_anchors:
                        for tgt_idx in self.idx_addr_anchors[k]:
                            addr_scores[tgt_idx] += 3.0
                            
            if numbers and phonetic_keys:
                for num in numbers[:2]:
                    for pk in phonetic_keys[:2]:
                        k = f"pn:{pk}_{num}"
                        if k in self.idx_addr_anchors:
                            for tgt_idx in self.idx_addr_anchors[k]:
                                addr_scores[tgt_idx] += 5.0
            
            all_indices = set(name_scores.keys()) | set(addr_scores.keys())
            if not all_indices:
                candidates_out[s1_id] = []
                continue
                
            # Dual-Evidence Boosted Scoring
            final_scores = {}
            for idx in all_indices:
                n_s = name_scores.get(idx, 0.0)
                a_s = addr_scores.get(idx, 0.0)
                if n_s > 0 and a_s > 0:
                    final_scores[idx] = (n_s + a_s) * 3.0 + (n_s * a_s)
                elif n_s > 0:
                    final_scores[idx] = n_s * 1.5
                else:
                    final_scores[idx] = a_s
                    
            # 1. Take top candidates by overall score
            top_main = sorted(final_scores.keys(), key=lambda idx: final_scores[idx], reverse=True)[:self.top_k - 6]
            selected = list(top_main)
            selected_set = set(selected)
            
            # 2. GUARANTEED NAME-RESERVE (Protects exact name matches with missing addresses!)
            if name_scores:
                top_name = sorted(name_scores.keys(), key=lambda idx: name_scores[idx], reverse=True)[:6]
                for n_idx in top_name:
                    if n_idx not in selected_set and len(selected) < self.top_k:
                        selected.append(n_idx)
                        selected_set.add(n_idx)
                        
            # Fill remaining budget if needed
            if len(selected) < self.top_k:
                rem = [idx for idx in sorted(final_scores.keys(), key=lambda idx: final_scores[idx], reverse=True) if idx not in selected_set]
                selected.extend(rem[:self.top_k - len(selected)])
                
            candidates_out[s1_id] = [self.target_ids[idx] for idx in selected[:self.top_k]]
            
        return candidates_out

def evaluate_candidate_recall(candidates: dict[str, list[str]], gt_path: str):
    gt_df = pl.read_csv(gt_path, separator="\t")
    
    total_true_links = 0
    recalled_links = 0
    candidate_counts = []
    
    for row in gt_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        if s1_id not in candidates:
            continue
            
        cand_list = set(candidates[s1_id])
        candidate_counts.append(len(cand_list))
        
        matches_str = row["matched_entity_ids"]
        if matches_str is not None and str(matches_str).strip():
            true_ids = [m.strip() for m in str(matches_str).split(",") if m.strip()]
            total_true_links += len(true_ids)
            for tid in true_ids:
                if tid in cand_list:
                    recalled_links += 1
                    
    recall_ceiling = (recalled_links / total_true_links * 100.0) if total_true_links > 0 else 0.0
    avg_cands = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
    
    print("\n" + "=" * 55)
    print("      CANDIDATE GENERATION BENCHMARK EVALUATION")
    print("=" * 55)
    print(f"Total True Ground Truth Matches: {total_true_links:,}")
    print(f"Recalled Matches in Candidate Set: {recalled_links:,}")
    print(f"Candidate Recall Ceiling:       {recall_ceiling:.2f}%")
    print(f"Average Candidates per S1:       {avg_cands:.2f}")
    print(f"Max Candidates Allowed:          50")
    print("=" * 55)
    return recall_ceiling, avg_cands

if __name__ == '__main__':
    val_dir = "dataset_val"
    print("Loading validation datasets...")
    s1_val = pl.read_csv(f"{val_dir}/val_source1.tsv", separator="\t")
    s2_val = pl.read_csv(f"{val_dir}/val_source2.tsv", separator="\t")
    s3_val = pl.read_csv(f"{val_dir}/val_source3.tsv", separator="\t")
    
    targets_all = pl.concat([s2_val, s3_val])
    print(f"Loaded {len(s1_val)} S1 validation rows and {len(targets_all)} target rows.")
    
    all_candidates = {}
    countries = s1_val["country"].unique().to_list()
    t_start = time.time()
    
    for country in countries:
        print(f"\n--- Processing Country: {country} ---")
        s1_c = s1_val.filter(pl.col("country") == country)
        tgt_c = targets_all.filter(pl.col("country") == country)
        
        cg = MultiPassCandidateGenerator(top_k=50)
        cg.build_target_index(tgt_c)
        cands_c = cg.generate_candidates_for_s1(s1_c)
        all_candidates.update(cands_c)
        
    t_elapsed = time.time() - t_start
    print(f"\nCompleted Candidate Generation for all countries in {t_elapsed:.2f} seconds!")
    evaluate_candidate_recall(all_candidates, f"{val_dir}/val_ground_truth.tsv")
