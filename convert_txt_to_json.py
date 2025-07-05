import json

#Read quotes from txt
with open("my_quotes.txt", "r", encoding="utf-8") as f:
    quotes = [line.strip() for line in f if line.strip()]

#Remove duplicates
unique_quotes = sorted(set(quotes))

#save the quotes to json file
with open("quotes.json", "w", encoding="utf-8") as f:
    json.dump(quotes, f, indent=4, ensure_ascii=False)

print(f"Successfully saved {len(quotes)} quotes to quotes.json")
