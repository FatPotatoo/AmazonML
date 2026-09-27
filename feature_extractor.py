import re
import sys
import numpy as np
import polars as pl
from rapidfuzz import fuzz, distance
from normalizer import normalize_text, parse_address_anchors

sys.stdout.reconfigure(encoding='utf-8')

def extract_features_for_pairs(pairs: list[dict], s1_dict: dict, tgt_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Given candidate pairs list [{'s1_id': ..., 'tgt_id': ..., 'block_score': ..., 'rank': ..., 'label': ...}],
    extract vectorized RapidFuzz similarity features.
    Returns (X, y)
    """
    n_samples = len(pairs)
    # Pre-allocate feature matrix (16 features)
    # 0: name_ratio
    # 1: name_token_sort
    # 2: name_token_set
    # 3: name_jaro_winkler
    # 4: name_partial_ratio
    # 5: name_len_diff
    # 6: name_exact_match
    # 7: addr_token_set
    # 8: addr_token_sort
    # 9: addr_jaro_winkler
    # 10: addr_num_jaccard
    # 11: addr_num_exact
    # 12: is_tgt_addr_empty
    # 13: name_x_addr
    # 14: block_score
    # 15: rank
    X = np.zeros((n_samples, 20), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int32)
    
    # Cache normalized strings to avoid redundant normalization
    norm_cache = {}
    def get_norm(text):
        if text not in norm_cache:
            norm_cache[text] = normalize_text(text)
        return norm_cache[text]
        
    # Cache address parsed data to avoid redundant work
    addr_cache = {}
    def get_addr_info(addr):
        if addr not in addr_cache:
            p = parse_address_anchors(addr)
            norm_a = get_norm(addr) if not p['is_empty'] else ""
            addr_cache[addr] = (set(p['numbers']), p['postal_code'], p['is_empty'], norm_a)
        return addr_cache[addr]

    for i in range(n_samples):
        if i > 0 and i % 250000 == 0:
            print(f"    Extracted {i:,} / {n_samples:,} pairs ({i*100//n_samples}%)...", flush=True)
        p = pairs[i]
        s1_id = p['s1_id']
        tgt_id = p['tgt_id']
        y[i] = p.get('label', 0)
        
        s1_rec = s1_dict[s1_id]
        tgt_rec = tgt_dict[tgt_id]
        
        # Names
        s1_name_clean = get_norm(s1_rec['business_name'])
        tgt_name_clean = get_norm(tgt_rec['business_name'])
        
        # RapidFuzz name metrics
        n_ratio = fuzz.ratio(s1_name_clean, tgt_name_clean) / 100.0
        n_tsort = fuzz.token_sort_ratio(s1_name_clean, tgt_name_clean) / 100.0
        n_tset = fuzz.token_set_ratio(s1_name_clean, tgt_name_clean) / 100.0
        n_jw = distance.JaroWinkler.similarity(s1_name_clean, tgt_name_clean)
        n_part = fuzz.partial_ratio(s1_name_clean, tgt_name_clean) / 100.0
        n_len_diff = abs(len(s1_name_clean) - len(tgt_name_clean))
        n_exact = 1.0 if (s1_name_clean and s1_name_clean == tgt_name_clean) else 0.0
        
        # Token metrics
        s1_toks = s1_name_clean.split()
        tgt_toks = tgt_name_clean.split()
        first_tok_match = 1.0 if (s1_toks and tgt_toks and s1_toks[0] == tgt_toks[0]) else 0.0
        shared_toks = float(len(set(s1_toks) & set(tgt_toks)))
        
        # Addresses
        s1_raw_addr = str(s1_rec['business_address'])
        tgt_raw_addr = str(tgt_rec['business_address'])
        
        s1_nums, s1_pcode, s1_empty, s1_addr_clean = get_addr_info(s1_raw_addr)
        tgt_nums, tgt_pcode, tgt_empty, tgt_addr_clean = get_addr_info(tgt_raw_addr)
        
        # Postal match signal
        if s1_pcode and tgt_pcode:
            if s1_pcode == tgt_pcode:
                postal_match = 1.0
                postal_prefix = 1.0
            else:
                postal_match = -1.0 # Hard location mismatch penalty!
                postal_prefix = 1.0 if s1_pcode[:3] == tgt_pcode[:3] else 0.0
        else:
            postal_match = 0.0
            postal_prefix = 0.0
        
        if tgt_empty or s1_empty:
            a_tset = 0.0
            a_tsort = 0.0
            a_jw = 0.0
            num_jaccard = 0.0
            num_exact = 0.0
            is_empty = 1.0
        else:
            a_tset = fuzz.token_set_ratio(s1_addr_clean, tgt_addr_clean) / 100.0
            a_tsort = fuzz.token_sort_ratio(s1_addr_clean, tgt_addr_clean) / 100.0
            a_jw = distance.JaroWinkler.similarity(s1_addr_clean, tgt_addr_clean)
            
            # Number overlap
            intersect = len(s1_nums & tgt_nums)
            union = len(s1_nums | tgt_nums)
            num_jaccard = (intersect / union) if union > 0 else 0.0
            num_exact = 1.0 if intersect > 0 else 0.0
            is_empty = 0.0
            
        # Interactions
        n_x_a = n_tset * a_tset if is_empty == 0.0 else n_tset * 0.5
        b_score = float(p.get('block_score', 0.0))
        rank = float(p.get('rank', 50))
        
        X[i, 0] = n_ratio
        X[i, 1] = n_tsort
        X[i, 2] = n_tset
        X[i, 3] = n_jw
        X[i, 4] = n_part
        X[i, 5] = n_len_diff
        X[i, 6] = n_exact
        X[i, 7] = first_tok_match
        X[i, 8] = shared_toks
        X[i, 9] = a_tset
        X[i, 10] = a_tsort
        X[i, 11] = a_jw
        X[i, 12] = num_jaccard
        X[i, 13] = num_exact
        X[i, 14] = postal_match
        X[i, 15] = postal_prefix
        X[i, 16] = is_empty
        X[i, 17] = n_x_a
        X[i, 18] = b_score
        X[i, 19] = rank
        
    return X, y

FEATURE_NAMES = [
    'name_ratio', 'name_token_sort', 'name_token_set', 'name_jaro_winkler',
    'name_partial_ratio', 'name_len_diff', 'name_exact_match',
    'first_token_match', 'shared_token_count',
    'addr_token_set', 'addr_token_sort', 'addr_jaro_winkler',
    'addr_num_jaccard', 'addr_num_exact', 
    'postal_match', 'postal_prefix_match', 'is_tgt_addr_empty',
    'name_x_addr', 'block_score', 'rank'
]
