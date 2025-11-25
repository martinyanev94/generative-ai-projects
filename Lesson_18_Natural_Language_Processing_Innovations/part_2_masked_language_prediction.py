from transformers import BertTokenizer, BertForMaskedLM
from transformers import pipeline

# Loading the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Input sentence
sentence = "The cat sat on the [MASK]"

# Tokenizing and encoding the input
input_ids = tokenizer.encode(sentence, return_tensors="pt")

# Performing masked language prediction
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs.logits

# Getting the most likely word for the masked token
predicted_index = predictions[0, 5].argmax(dim=-1).item()
predicted_token = tokenizer.decode(predicted_index)

print(f'The masked word is predicted to be: {predicted_token}')
