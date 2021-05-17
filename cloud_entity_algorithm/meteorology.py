"""
Collection of meteorology functions
"""


def wind_profile_power_func(u_r, z_r, power=0.11):
    """
    Wind profile power law

    Input
    -----
    u_r : float
        windspeed at reference height
    z_r : float
        reference height
    """
    f = lambda h: u_r * (h / z_r) ** power
    return f
