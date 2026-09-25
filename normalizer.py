import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
from unidecode import unidecode
import jellyfish

LEGAL_SUFFIXES = {
    'pvt', 'ltd', 'limited', 'private', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'sarl', 'sas', 'sasu', 'sa', 'eurl', 'sci', 'snc', 'co', 'company',
    'enterprises', 'enterprise', 'industries', 'industry', 'services', 'service',
    'associates', 'group', 'holdings'
}

RE_LEGAL = re.compile(r'\b(' + '|'.join(LEGAL_SUFFIXES) + r')\b', re.IGNORECASE)
RE_CONSECUTIVE_LETTERS = re.compile(r'([a-z])\1+')
RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')
RE_SPACES = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\d+')
RE_POSTAL_CODE = re.compile(r'\b\d{5,6}\b')

GENERIC_ADDR_WORDS = {
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln',
    'drive', 'dr', 'block', 'near', 'opp', 'floor', 'unit', 'apt',
    'suite', 'building', 'bldg', 'rue', 'boulevard', 'blvd', 'terrace',
    'court', 'ct', 'highway', 'hwy', 'parkway', 'pkwy', 'way', 'place',
    'null', 'none'
}

def normalize_text(text: str) -> str:
    """Normalize text: transliterate Indic/French, collapse acronym dots, strip legal suffixes."""
    if not text or str(text) == 'nan' or str(text) == 'None':
        return ''
    
    # Transliterate to ASCII
    t = unidecode(str(text)).lower()
    
    # Fold dot-separated acronyms (e.g., l.l.c. -> llc, p.o. -> po, u.s. -> us)
    t = re.sub(r'\b([a-z])\.([a-z])\.([a-z])\b', r'\1\2\3', t)
    t = re.sub(r'\b([a-z])\.([a-z])\b', r'\1\2', t)
    
    # Strip domain extensions
    t = re.sub(r'\b(www\.|https?://|\.com|\.in|\.org|\.net|\.co|\.fr)\b', ' ', t)
    
    # Strip legal suffixes
    t = RE_LEGAL.sub(' ', t)
    
    # Collapse consecutive repeated letters (raam -> ram, innnvesttmenntts -> investments)
    t = RE_CONSECUTIVE_LETTERS.sub(r'\1', t)
    
    # Strip non-alphanumeric
    t = RE_NON_ALPHANUM.sub(' ', t)
    
    return RE_SPACES.sub(' ', t).strip()

def extract_name_tokens(clean_name: str) -> list[str]:
    """Extract significant name tokens (length >= 2)."""
    if not clean_name:
        return []
    tokens = clean_name.split()
    return [tok for tok in tokens if len(tok) >= 2]

def extract_prefix_shingles(tokens: list[str], min_len: int = 3, max_len: int = 4) -> list[str]:
    """Extract 3-4 character prefix shingles."""
    shingles = []
    for tok in tokens:
        if len(tok) >= min_len:
            shingles.append(tok[:min_len])
            if len(tok) >= max_len:
                shingles.append(tok[:max_len])
    return list(set(shingles))

def extract_phonetic_keys(tokens: list[str]) -> list[str]:
    """Extract Soundex and Metaphone keys for tokens >= 3 chars."""
    p_keys = []
    for tok in tokens:
        if len(tok) >= 3 and not tok.isdigit():
            sx = jellyfish.soundex(tok)
            if sx:
                p_keys.append(f"sx:{sx}")
            mp = jellyfish.metaphone(tok)
            if mp and len(mp) >= 2:
                p_keys.append(f"mp:{mp}")
    return list(set(p_keys))

def parse_address_anchors(raw_address: str) -> dict:
    """Extract street numbers, postal codes, and content words."""
    if not raw_address or str(raw_address) == 'nan' or str(raw_address) == 'None':
        return {
            'numbers': [],
            'postal_code': None,
            'clean_address_tokens': [],
            'is_empty': True
        }
        
    clean_addr = normalize_text(raw_address)
    tokens = clean_addr.split()
    
    postal_codes = RE_POSTAL_CODE.findall(clean_addr)
    postal_code = postal_codes[0] if postal_codes else None
    
    raw_nums = RE_NUMBERS.findall(str(raw_address))
    numbers = [num for num in raw_nums if 1 <= len(num) <= 6]
    
    addr_tokens = [
        tok for tok in tokens 
        if len(tok) >= 3 and tok not in GENERIC_ADDR_WORDS and not tok.isdigit()
    ]
    
    return {
        'numbers': numbers[:3],
        'postal_code': postal_code,
        'clean_address_tokens': addr_tokens,
        'is_empty': False
    }
