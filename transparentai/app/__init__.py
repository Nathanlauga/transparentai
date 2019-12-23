import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from flask import Flask, request
from flask_babel import Babel

app = Flask(__name__, instance_relative_config=True)
babel = Babel(app)

app.config.from_object("config")

@babel.localeselector
def get_locale():
    print(request.accept_languages.best_match(app.config['LANGUAGES']))
    return request.accept_languages.best_match(app.config['LANGUAGES'])


from app import views