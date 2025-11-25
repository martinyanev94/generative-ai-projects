def filter_bias(data, bias_indicator):
    return [item for item in data if bias_indicator not in item]

filtered_data = filter_bias(original_data, "biased_term")
