from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define a prompt
prompt = "Once upon a time in a magical forest"

# Encode and generate text
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(f'Generated text: {generated_text}')
