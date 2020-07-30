""" UNIT OF MEASUREMENT CONVERSIONS 

Source code from https://github.com/responsibleproblemsolving/energy-usage/blob/master/energyusage/convert.py
"""


def to_joules(ujoules):
    """ Converts from microjoules to joules """
    return ujoules*10**(-6)


def to_kwh(joules):
    """ Converts from watts used in a timeframe (in hours) to kwh """
    watt_hours = joules
    return watt_hours / 1000


def to_MWh(kwh):
    """ Converts from kilowatt-hours to megawatt-hours """
    return (kwh / 1000)


def kwh_to_mmbtu(kwh):
    ''' Convert from kilowatt hour to million British thermal unit '''
    # https://en.wikipedia.org/wiki/British_thermal_unit#As_a_unit_of_power
    return kwh * 0.003412142


def coal_to_carbon(kwh):
    '''
    2195.2 lbs CO2/MWh
    source: reverse-engineered from eGRID data
    '''
    MWh = to_MWh(kwh)
    lbs_carbon = 2195.2 * MWh
    return lbs_to_kgs(lbs_carbon)


def natural_gas_to_carbon(kwh):
    '''
    1639.89 lbs CO2/MWh
    source: reverse-engineered from eGRID data
    '''
    MWh = to_MWh(kwh)
    lbs_carbon = 1639.89 * MWh
    return lbs_to_kgs(lbs_carbon)


def petroleum_to_carbon(kwh):
    '''
    Oil: 1800.49 lbs CO2/MWh
    source: reverse-engineered from eGRID data
    '''
    MWh = to_MWh(kwh)
    lbs_carbon = 1800.49 * MWh
    return lbs_to_kgs(lbs_carbon)


def lbs_to_kgs(lbs):
    '''Convert from pounds to kilograms'''
    return lbs * 0.45359237


""" CARBON EQUIVALENCY """


def carbon_to_miles(kg_carbon):
    '''
    8.89 × 10-3 metric tons CO2/gallon gasoline ×
    1/22.0 miles per gallon car/truck average ×
    1 CO2, CH4, and N2O/0.988 CO2 = 4.09 x 10-4 metric tons CO2E/mile
    Source: EPA
    '''
    return 4.09 * 10**(-7) * kg_carbon # number of miles driven by avg car


def carbon_to_home(kg_carbon):
    '''
    Total CO2 emissions for energy use per home: 5.734 metric tons CO2 for electricity
    + 2.06 metric tons CO2 for natural gas + 0.26 metric tons CO2 for liquid petroleum gas
     + 0.30 metric tons CO2 for fuel oil  = 8.35 metric tons CO2 per home per year / 52 weeks
     = 160.58 kg CO2/week on average
    Source: EPA
    '''
    return kg_carbon * 10**(-3) / 8.35 / 52 / 7 #percent of CO2 used in an avg US household in a week


def carbon_to_tv(kg_carbon):
    '''
    Gives the amount of minutes of watching a 32-inch LCD flat screen tv required to emit and
    equivalent amount of carbon. Ratio is 0.097 kg CO2 / 1 hour tv
    '''
    return kg_carbon * (1 / .097) * 60
