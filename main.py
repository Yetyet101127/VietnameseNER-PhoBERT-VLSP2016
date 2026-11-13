from transformers import pipeline

ner = pipeline(
    "ner",
    model="./best_model",
    tokenizer="./best_model",
    aggregation_strategy="none",
)

while True:
    text = input("Input (<=256 tokens): ").strip()
    if not text:
        break
    for i in ner(text):
        print(f"{i['word']:<20} {i['entity']:<8} {i['score']:.3f}")
    print()