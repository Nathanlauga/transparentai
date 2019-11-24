import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from flask import Flask

# Initialize the app
app = Flask(__name__, instance_relative_config=True)

# Load the config file
app.config.from_object("config")

# Load the views
from app import views