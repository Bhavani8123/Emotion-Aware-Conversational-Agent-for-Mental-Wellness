from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise','Depression','Anxiety','Stress']

def detect_mental_labels(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0].numpy()
    return [(label, round(float(prob) * 100, 2)) for label, prob in zip(labels, probs) if prob >= threshold]

# 🧪 Test:
print(detect_mental_labels("I feel very sad and alone"))
