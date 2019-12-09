import os
from .data import load_questions

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

questions = load_questions(path=path)

