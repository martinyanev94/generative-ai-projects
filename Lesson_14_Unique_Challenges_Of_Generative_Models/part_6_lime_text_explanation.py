from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=["negative", "positive"])
idx = 1  # Arbitrary instance index from the dataset
exp = explainer.explain_instance(data[idx], model.predict_proba, top_labels=1)
exp.show_in_notebook(text=True)
