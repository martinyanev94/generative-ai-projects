from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

sentence = "The quick brown fox jumps over the [MASK] dog."
inputs = tokenizer(sentence, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
predictions = outputs.logits
masked_index = inputs['input_ids'][0].tolist().index(tokenizer.mask_token_id)
predicted_index = predictions[0, masked_index].argmax().item()
predicted_token = tokenizer.decode(predicted_index)

print(f"Predicted token: {predicted_token}")
