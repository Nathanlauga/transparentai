from flask import render_template
from app import app


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

# @app.errorhandler(404)
# def not_found(e):
#     return render_template("404.html")
