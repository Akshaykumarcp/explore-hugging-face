# CREDITS: https://huggingface.co/course/chapter1/3?fw=pt

from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)

# {'score': 0.6385916471481323, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}