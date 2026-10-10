import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent / "BertForSequenceClassification"


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "This is a sample sentence for ezkl."

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    predicted_class = int(torch.argmax(logits, dim=-1).item())

    print(f"Input text     : {text}")
    print(f"Input ids      : {inputs['input_ids'].tolist()}")
    print(f"Logits         : {logits.tolist()}")
    print(f"Probabilities  : {probs.tolist()}")
    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    main()
