def collect_feedback(predicted_text, user_input):
    feedback = input(f"Is this text appropriate: '{predicted_text}'? (yes/no) ")
    if feedback.lower() == 'no':
        # Code to adjust the training data/strategy accordingly
        print("Adjusting the model based on user feedback.")
