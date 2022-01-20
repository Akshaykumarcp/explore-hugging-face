# CREDITS: https://huggingface.co/course/chapter1/3?fw=pt

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# sentiment analysis using single sentence
classifier("I've been waiting for a HuggingFace course my whole life.")
# [{'label': 'POSITIVE', 'score': 0.9598046541213989}]

# sentiment analysis using multiple sentences
classifier(["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"])
# [{'label': 'POSITIVE', 'score': 0.9598046541213989}, {'label': 'NEGATIVE', 'score': 0.9994558691978455}]

