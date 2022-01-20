# CREDITS: https://huggingface.co/course/chapter1/3?fw=pt

from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")

# [{'translation_text': 'This course is produced by Hugging Face.'}]