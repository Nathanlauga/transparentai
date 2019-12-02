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

@app.route('/ai/user-needs/', methods=['GET', 'POST'])
def user_needs():
    title = '1. User needs'
    sections = questions['section'].unique()
    return render_template("ai/user-needs.html", title=title, sections=sections, questions=questions)

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
