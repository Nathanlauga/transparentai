from .db import DB
from .db import format_anwser_for_db

DB = DB()
questions = DB.get_questions()
