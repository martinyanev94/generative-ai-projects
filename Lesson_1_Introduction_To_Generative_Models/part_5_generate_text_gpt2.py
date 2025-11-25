from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Generate text
prompt = "Once upon a time in a land far away"
inputs = tokenizer.encode(prompt, return_tensors='pt')

# Generate text
outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
