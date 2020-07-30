import subprocess


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
