import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from app import app

def is_path_format_correct(path: str):
    """
    Control if path finish with a '/'

    Parameters
    ----------
    path : str
        path to control

    Returns
    -------
    bool:
        whether the path finish with a '/' or not
    """
    return path[-1] == '/'


def remove_empty_from_dict(d: dict):
    """
    Remove all None and '' values from a given dictionnary

    Parameters
    ----------
    d : dict
        dictionnary to check

    Returns
    -------
    dict:
        dictionnary without None or '' values
    """
    return {k: v for k, v in d.items() if v not in [None, '', [], {}]}


def get_locale_language(lang_header: str):
    """
    Return the language based on request header object

    Parameters
    ----------
    header: dict
        Accept-Language attribute from headers retrieve 
        with request.headers object
    
    Returns
    -------
    str:
        language that is valid for this app 
    """
    lang = lang_header.split(',')

    if len(lang) > 1:
        lang = lang[1].split(';')
    
    lang = lang[0].strip()
    lang = lang if lang in app.config['ACCEPTED_LANG'] else app.config['DEFAULT_LANG']

    return lang