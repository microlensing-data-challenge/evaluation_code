# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:33:04 2018

@author: rstreet
"""

import parse_table1
import astropy.units as u
import numpy as np

def test_calc_orbital_parameters():
    """Function to test the calculation of the orbital parameterization
    from the circular orbital parameters"""
    
    # Angles defined in degrees
    phase = 20.0 * (np.pi/180.0) # deg -> rads
    inc = 20.0 * (np.pi/180.0) # deg -> rads
    period = 1.0 * 365.24   # years -> days
    RE = 3.06162            # AU
    a = 1.0 / RE            # AU -> thetaE
    q = 1.0
    alpha0 = 10.0 * (np.pi/180.0)  # deg -> rads
    t_ref = 2458234.0       # days
    t = t_ref - 1.0         # days
    
    (dsdt, dalphadt) = parse_table1.calc_orbital_parameters(phase, inc, 
                                                            period, a, q, 
                                                            alpha0, t_ref, t,
                                                            verbose=True)

    print('ds/dt = '+str(dsdt)+' thetaE/year')
    print('dalpha/dt = '+str(dalphadt)+' radians/year')
    
if __name__ == '__main__':
    
    test_calc_orbital_parameters()
    