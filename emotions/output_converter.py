import re
import random
from collections import Counter
from emotions.emoji_data import emoji_dict

DEFAULT_EMOJI_POOL = ("🙂", "✨", "🎉", "💡", "🚀", "🌟", "🧠", "🔥")

def _flatten_emoji_values(values):
    flat = []
    for v in values:
        if isinstance(v, str):
            flat.append(v)
        elif isinstance(v, (list, tuple, set)):
            flat.extend([e for e in v if isinstance(e, str) and e])
    return flat

def add_emoji_to_output(output: str) -> str:
    # output'u küçük harfe çevir ve kelimeleri ayır
    words = re.findall(r'\b\w+\b', output.lower())

    # kelimeler ve emoji_dict anahtarlarını kesiştir
    matched = [word for word in words if word in emoji_dict]

    if matched:
        # en sık geçen kelimeyi bul
        most_common_word, _ = Counter(matched).most_common(1)[0]
        value = emoji_dict[most_common_word]
        # Değer liste ise ilkini al; string ise direkt kullan
        emoji = value[0] if isinstance(value, (list, tuple)) and value else value
        # Her ihtimale karşı geçersizse fallback havuzdan seç
        if not isinstance(emoji, str) or not emoji:
            pool = _flatten_emoji_values(emoji_dict.values()) or list(DEFAULT_EMOJI_POOL)
            emoji = random.choice(pool)
        return output + " " + emoji

    # Eşleşme yoksa rastgele bir emoji ekle
    pool = _flatten_emoji_values(emoji_dict.values())
    if not pool:
        pool = list(DEFAULT_EMOJI_POOL)
    emoji = random.choice(pool)
    return output + " " + emoji