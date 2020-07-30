import energyusage


def evaluate_kWh(func, *args, verbose=False):
    """Using energyusage.evaluate function returns
    the result of the function and the effective 
    emissions of the function (in kWh)

    With verbose = True you can see the report with details.

    If you want a pdf please use the following:

    >>> energyusage.evaluate(func, *args, pdf=True)

    From energyusage_ package.

    .. _energyusage: https://github.com/responsibleproblemsolving/energy-usage/tree/master/energyusage

    Parameters
    ----------
    func: 
        User's function
    verbose: bool (default False)
        Whether it shows details or not

    Returns
    -------
    float:
        effective emissions of the function in kWh
    any
        function's return
    """

    _, kWh, fun_return = energyusage.evaluate(func, *args, energyOutput=True,
                                              pdf=False, printToScreen=verbose)

    return kWh, fun_return
