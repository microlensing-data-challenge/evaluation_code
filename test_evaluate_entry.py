# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:27:12 2018

@author: rstreet
"""

import evaluate_entry

def test_compare_parameter():

    true_par = 2459815.21264422
    fitted_par = 2459816.21546
    fitted_error = 0.0016982558
    dpar = true_par - fitted_par
    
    (delta_par, within_1sig, within_3sig) = evaluate_entry.compare_parameter(true_par,fitted_par,fitted_error)
    
    assert(delta_par == dpar)
    assert(within_1sig == False)
    assert(within_3sig == False)

    true_par = 2459815.2126
    fitted_par = 2459815.212605
    fitted_error = 0.001
    dpar = true_par - fitted_par
    
    (delta_par, within_1sig, within_3sig) = evaluate_entry.compare_parameter(true_par,fitted_par,fitted_error)
    
    assert(delta_par == dpar)
    assert(within_1sig == True)
    assert(within_3sig == True)

    true_par = 2459815.2126
    fitted_par = 2459815.2106
    fitted_error = 0.001
    dpar = true_par - fitted_par
    
    (delta_par, within_1sig, within_3sig) = evaluate_entry.compare_parameter(true_par,fitted_par,fitted_error)
    
    assert(delta_par == dpar)
    assert(within_1sig == False)
    assert(within_3sig == True)


if __name__ == '__main__':
    
    test_compare_parameter()
    