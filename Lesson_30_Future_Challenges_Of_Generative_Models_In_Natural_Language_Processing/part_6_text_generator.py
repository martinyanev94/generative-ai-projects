def generate_text(model, input_seq, filter_words):
    generated_sequence = model(input_seq)
    generated_text = decode(generated_sequence)  # Assuming decode is a function that converts tensor to text
    
    for word in filter_words:
        if word in generated_text:
            raise ValueError("Generated text contains filtered content")
    
    return generated_text
