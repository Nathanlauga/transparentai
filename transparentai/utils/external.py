import energyusage
import subprocess


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


def check_packages_security(full_report=True):
    """Using safety package, check out the known vulnerabilities 
    of the installed packages.

    For more details you can look at the package page :
    https://github.com/pyupio/safety

    Parameters
    ----------
    full_report: True
        Whether you want the full report or 
        short report.    
    """
    report = '--full-report' if full_report else '--short-report'
    result = subprocess.run(
        ['safety', 'check', report], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
