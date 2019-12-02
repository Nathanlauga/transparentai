from flask import render_template
from app import app

from data import questions


@app.route('/', methods=['GET', 'POST'])
def index():
    title = 'Home'
    return render_template("index.html", title=title)


@app.route('/ai/', methods=['GET', 'POST'])
def ai():
    title = 'Create a new AI'
    sections = questions['section'].unique()
    return render_template("ai.html", title=title, sections=sections, questions=questions)


@app.route('/glossary/', methods=['GET', 'POST'])
def glossary():
    title = 'Glossary'
    return render_template("glossary.html", title=title)


@app.route('/ressources/', methods=['GET', 'POST'])
def ressources():
    title = 'Ressources'
    return render_template("ressources.html", title=title)


@app.errorhandler(404)
def not_found(e):
    title = 'Page not found'
    return render_template("404.html", title=title)
