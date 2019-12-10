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