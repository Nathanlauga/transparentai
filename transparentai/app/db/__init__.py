from .db import DB


db = DB()
questions = db.get_questions()
