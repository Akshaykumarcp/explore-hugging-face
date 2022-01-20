# CREDITS: https://huggingface.co/course/chapter1/3?fw=pt

from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")

""" 
[{'generated_text': 'In this course, we will teach you how to understand and use '
                    'data flow and data interchange when handling user data. We '
                    'will be working with one or more of the most commonly used '
                    'data flows — data flows of various types, as seen by the '
                    'HTTP'}] """