import json
from sentence_transformers import SentenceTransformer, util # type: ignore

# Tags you want your quotes to be classified under
TAG_POOL = [
        "happiness", "sadness", "peace", "stress", "loneliness", "confidence", "hope", "anger", "grief", "anxiety", "fear", "joy", "contentment",
        "motivation", "discipline", "consistency", "self-control", "focus", "purpose", "ambition", "resilience", "growth", "habits", "mindset",
        "life", "death", "truth", "meaning", "freedom", "destiny", "acceptance", "regret", "change", "choices", "wisdom", "time", "reflection",
        "love", "compassion", "empathy", "friendship", "family", "connection", "kindness", "forgiveness", "trust", "heartbreak",
        "money", "business", "success", "leadership", "risk", "entrepreneurship", "opportunity", "value", "productivity",
        "spirituality", "karma", "consciousness", "mindfulness", "humility", "gratitude", "faith", "ethics", "balance",
        "nature", "simplicity", "beauty", "calm", "environment", "animals", "trees", "seasons", "universe", "earth", "space",
        "lust", "desire", "greed", "addiction", "ego", "materialism", "envy", "anger", "jealousy", "temptation"

]


# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load quotes from text file
with open("my_quotes.txt", "r", encoding="utf-8") as f:
    quotes = [line.strip() for line in f if line.strip()]

# Encode all tags once
tag_embeddings = model.encode(TAG_POOL, convert_to_tensor=True)

# Tag each quote
quote_data = []
for quote in quotes:
    quote_embedding = model.encode(quote, convert_to_tensor=True)
    similarities = util.cos_sim(quote_embedding, tag_embeddings)[0]
    top_tags = [TAG_POOL[i] for i in similarities.argsort(descending=True)[:6]]
    
    quote_data.append({
        "quote": quote,
        "tags": top_tags
    })

# Save to JSON
with open("quotes.json", "w", encoding="utf-8") as f:
    json.dump(quote_data, f, indent=4, ensure_ascii=False)

print(f"✅ Tagged and saved {len(quote_data)} quotes to quotes.json")
