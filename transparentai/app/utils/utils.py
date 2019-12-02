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