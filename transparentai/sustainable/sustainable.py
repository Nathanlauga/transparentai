import json
import energyusage.evaluate
from . import convert
from os.path import dirname, abspath

def get_energy_data(year=2016):
    """Loads enery data from a specify year (only 2016 is currently available)

    Parameters
    ----------
    year: int (default 2016)
        Year of the energy mix data

    Returns
    -------
    dict:
        Energy mix per country of the selected year
    """
    path = dirname(dirname(abspath(__file__)))

    with open(f'{path}/src/energy{str(year)}.json', 'r') as file:
        data = json.loads(file.read())
    file.close()

    return data


def energy_mix(location):
    """ Gets the energy mix information for a specific location

    Parameters
    ----------
    location: str
        user's location
    location_of_default: str
        Specifies which average to use if
        location cannot be determined

    Returns
    -------
    list: 
        percentages of each energy type

    Raises
    ------
    ValueError:
        location must be a valid countries
    """
    data = get_energy_data()
    valid_countries = list(data.keys())

    if location not in valid_countries:
        raise ValueError(
            'location must be one of the following countries: '+','.join(valid_countries))

    c = data[location]  # get country
    total, breakdown = c['total'], [c['coal'], c['petroleum'],
                                    c['naturalGas'], c['lowCarbon']]

    # Get percentages
    if total != 0:
        breakdown = list(map(lambda x: 100*x/total, breakdown))

    return breakdown


def emissions(process_kwh, breakdown, location):
    """ Calculates the CO2 emitted by the program based on the location

    Parameters
    ----------
    process_kwh: int 
        kWhs used by the process
    breakdown: list
        energy mix corresponding to user's location
    location: str
        location of user

    Returns
    -------
    float
        emission in kilograms of CO2 emitted

    Raises
    ------
    ValueError:
        Process wattage must be greater than 0.
    """

    if process_kwh < 0:
        raise ValueError("Process wattage must be greater than 0.")

    # Breaking down energy mix
    coal, petroleum, natural_gas, low_carbon = breakdown
    breakdown = [convert.coal_to_carbon(process_kwh * coal/100),
                 convert.petroleum_to_carbon(process_kwh * petroleum/100),
                 convert.natural_gas_to_carbon(process_kwh * natural_gas/100), 0]
    emission = sum(breakdown)

    return emission


def estimate_co2(hours, location, watts=250, powerLoss=0.8):
    """ Returns co2 consumption in kg CO2

    To find out the wattage of the machine used for training, I recommend you use
    this website: `Newegg's Power Supply Calculator`_ .

    Based on this website: `Power Management Statistics`_
    we can estimate an average wattage to be 250 Watts, but be carefull, it's
    only an estimation. So if you're using a computer with GPU or others components
    I recommend you use the first website that allows you to compute your wattage.

    .. _`Newegg's Power Supply Calculator`: https://www.newegg.com/tools/power-supply-calculator
    .. _`Power Management Statistics`: https://www.it.northwestern.edu/hardware/eco/stats.html


    Parameters
    ----------
    hours: int 
        time of training in hours
    location: str
        location of user
    watts: int (default 250)
        Wattage of the computer or server that
        was used for training
    powerLoss: float (default 0.8)
        PSU efficiency rating

    Returns
    -------
    float
        emission in kilograms of CO2 emitted
    """
    process_kwh = convert.to_kwh(watts*hours) / powerLoss
    breakdown = energy_mix(location)

    return emissions(process_kwh, breakdown, location)


# watts = 358
# hours = 8
# powerLoss = 0.8
# locations = ['France', 'United States']

# data = get_data()
# # locations = list(data.keys())
# # locations = [l for l in locations if l not in ['_define']]

# for location in locations:
#     co2 = estimate_co2(watts, hours, location)
#     print(location, co2)
