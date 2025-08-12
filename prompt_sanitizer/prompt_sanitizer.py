# python
import difflib
import re
import unicodedata
from typing import List, Tuple, Dict, Optional

# word_to_id: mevcut vocab sözlüğünüzü buraya verin
# Örn: word_to_id = {...}
# id_to_word = {v: k for k, v in word_to_id.items()}

def _simple_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)   # noktalama temizle
    text = re.sub(r"_+", " ", text)        # alt çizgileri boşluğa çevir
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    # Noktalama ve gereksiz karakterleri boşluğa çevir
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # Alt çizgiyi de gereksiz sayıyorsanız kaldırın
    text = re.sub(r"_+", " ", text)
    # Çoklu boşluk sıkıştırma
    text = re.sub(r"\s+", " ", text).strip()
    return text

def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = prev[j] + 1
            dele = cur[j - 1] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def normalized_similarity(a: str, b: str) -> float:
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(a, b) / m)

def build_first_char_index(vocab_words: List[str]) -> Dict[str, List[str]]:
    index = {}
    for w in vocab_words:
        if not w:
            continue
        index.setdefault(w[0], []).append(w)
    return index

def best_vocab_match(
    token: str,
    vocab_words: List[str],
    index_by_first_char: Optional[Dict[str, List[str]]] = None,
    threshold: float = 0.2
) -> Optional[str]:
    # Doğrudan eşleşme varsa hızlı dönüş
    if token in vocab_words:
        return token

    # Adayları daralt (aynı ilk harf + uzunluk yakınlığı)
    candidates = []
    if index_by_first_char is not None and token:
        candidates = index_by_first_char.get(token[0], [])
    if not candidates:
        candidates = vocab_words

    best_word = None
    best_sim = -1.0
    tlen = len(token)

    for w in candidates:
        # Basit uzunluk filtresi (performans için)
        if abs(len(w) - tlen) > max(2, int(0.6 * tlen)):
            continue
        sim = normalized_similarity(token, w)
        if sim > best_sim:
            best_sim = sim
            best_word = w

    if best_sim >= threshold:
        return best_word
    return None

def normalize_and_map_prompt(prompt: str, word_to_id: dict, sim_threshold: float = 0.7):
    """
    prompt içindeki kelimeleri vocab'e (word_to_id) eşler.
    - Doğrudan bulunan kelime -> eklenir
    - Bulunamazsa difflib ile en yakın kelime eşlenir (eşik üstündeyse)
    Her durumda (mapped_words, mapped_ids) döndürür; boş da olabilir ama asla None döndürmez.
    """
    if not isinstance(prompt, str):
        return [], []

    text = _simple_normalize(prompt)
    if not text:
        return [], []

    tokens = text.split()
    if not tokens:
        return [], []

    mapped_words = []
    mapped_ids = []

    vocab_words = list(word_to_id.keys())
    for tok in tokens:
        if tok in word_to_id:
            mapped_words.append(tok)
            mapped_ids.append(word_to_id[tok])
            continue

        # Yakın eşleme (difflib ratio 0..1 arası; genelde 0.6-0.8 iyi sonuç verir)
        best_word = None
        best_score = 0.0
        for vw in vocab_words:
            score = difflib.SequenceMatcher(a=tok, b=vw).ratio()
            if score > best_score:
                best_score = score
                best_word = vw

        if best_word is not None and best_score >= sim_threshold:
            mapped_words.append(best_word)
            mapped_ids.append(word_to_id[best_word])
        # Eşik altında ise bu token’ı atla

    return mapped_words, mapped_ids
