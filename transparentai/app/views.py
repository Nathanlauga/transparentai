from flask import render_template, request, redirect, url_for
from app import app
import pandas as pd

from flask_babel import _

from db import questions, DB
import db
import utils


@app.route('/', methods=['GET', 'POST'])
def index():
    title = _('Home')
    return render_template("index.html", title=title)


@app.route('/ai/', methods=['GET', 'POST'])
def ai():
    title = 'Create a new AI'
    sections = questions['section'].unique()
    return render_template("ai.html", title=title, sections=sections, questions=questions)


@app.route('/ai/define-ai/', methods=['GET', 'POST'])
def define_ai():
    if not DB.is_ai_in_creation():
        DB.init_ai()
    
    current_ai = DB.get_ai_in_creation()[0]
    ai_id = current_ai.doc_id

    return define_ai_id(id=ai_id)

@app.route('/ai/define-ai/<id>', methods=['GET', 'POST'])
def define_ai_id(id):
    if not DB.ai_exists(ai_id=int(id)):
        return redirect(url_for('ai'))

    title = '1. Define the AI'
    sections = questions['section'].unique()

    if request.method == 'POST':
        data = request.form.to_dict()
        data = db.format_anwser_for_db(answers=data)

        DB.add_answer_ai(ai_id=id, answers=data)

    answers = DB.get_answers_ai(ai_id=int(id))

    return render_template("ai/define-ai.html", title=title, sections=sections, questions=questions, answers=answers)

@app.route('/ai/data-collect/', methods=['GET', 'POST'])
def data_collect():
    title = '2. Data collection'
    return render_template("ai/data-collect.html", title=title)

@app.route('/ai/data-quality/', methods=['GET', 'POST'])
def data_quality():
    title = '3. Data quality'
    return render_template("ai/data-quality.html", title=title)
    
@app.route('/ai/eda/', methods=['GET', 'POST'])
def eda():
    title = '4. Exploratory data analysis'
    return render_template("ai/eda.html", title=title)

    
@app.route('/ai/data-preparation/', methods=['GET', 'POST'])
def data_preparation():
    title = '5. Data preparation'
    return render_template("ai/data-preparation.html", title=title)

    
@app.route('/ai/model-creation/', methods=['GET', 'POST'])
def model_creation():
    title = '6. Model creation'
    return render_template("ai/model-creation.html", title=title)

    
@app.route('/ai/model-tuning/', methods=['GET', 'POST'])
def model_tuning():
    title = '7. Model testing'
    return render_template("ai/model-tuning.html", title=title)

    
@app.route('/ai/model-deployment/', methods=['GET', 'POST'])
def model_deployment():
    title = '8. Model deployment'
    return render_template("ai/model-deployment.html", title=title)

    
@app.route('/ai/monitoring/', methods=['GET', 'POST'])
def model_monitoring():
    title = '9. Monitoring'
    return render_template("ai/monitoring.html", title=title)


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
