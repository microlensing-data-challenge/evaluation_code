# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:17:52 2018

@author: rstreet
"""

from sys import argv
from os import path, remove
import parse_table1
import logging
import matplotlib.pyplot as plt
import numpy as np

def evaluate_entry():
    """Function to evaluate the numerical data from a data challenge entry, 
    by comparing the fitted numerical parameters with the master table data"""
        
    params = get_args()
    
    log = start_log( params['log_file'] )
    
    master_data = parse_table1.read_master_table(params['master_file'])
    
    entry_data = parse_table1.read_standard_ascii_DC_table(params['entry_file'])
    
    check_classifications(params, master_data, entry_data, log)
    
    log.info( 'Analysis complete\n' )
    logging.shutdown()
    
def start_log(log_file):
    """Function to initialize a log file
    The new file will automatically overwrite any previously-existing logfile
    for the given reduction.  

    This function also configures the log file to provide timestamps for 
    all entries.  
    
    Returns:
        log       open logger object
    """
    
    # Console output not captured, though code remains for testing purposes
    console = False

    log_file = path.join(log_file)
    
    if path.isfile(log_file) == True:
        remove(log_file)
        
    # To capture the logging stream from the whole script, create
    # a log instance together with a console handler.  
    # Set formatting as appropriate.
    log = logging.getLogger( 'evaluation' )
    
    if len(log.handlers) == 0:
        log.setLevel( logging.INFO )
        file_handler = logging.FileHandler( log_file )
        file_handler.setLevel( logging.INFO )
        
        if console == True:
            console_handler = logging.StreamHandler()
            console_handler.setLevel( logging.INFO )
    
        formatter = logging.Formatter( fmt='%(asctime)s %(message)s', \
                                    datefmt='%Y-%m-%dT%H:%M:%S' )
        file_handler.setFormatter( formatter )

        if console == True:        
            console_handler.setFormatter( formatter )
    
        log.addHandler( file_handler )
        if console == True:            
            log.addHandler( console_handler )
    
    log.info('Starting analysis\n')
    
    return log

def get_args():
    """Function to request the required input parameters"""
    
    params = {}
    
    if len(argv) == 1:
        
        params['master_file'] = input('Please enter the path to the master parameter file: ')
        params['entry_file'] = input('Please enter the path to the entry data file: ')
        
    else:
        
        params['master_file'] = argv[1]
        params['entry_file'] = argv[2]
    
    params['log_dir'] = path.dirname(params['entry_file'])
    params['log_file'] = path.join(params['log_dir'], 'evaluation.log')
        
    return params

def check_classifications(params, master_data, entry_data, log):
    """Function to check whether models have been correctly classified"""
    
    log.info('Checking model classifications')
    
    # For each class, record [ n_good, n_bad ] classifications
    classes = { 'PSPL': {'n_good': 0, 'n_bad': 0, 'names': ['PSPL']},
                'Binary_star': {'n_good': 0, 'n_bad': 0, 'names': ['USBL']},
                'Binary_planet': {'n_good': 0, 'n_bad': 0, 'names': ['USBL']},
                'CV': {'n_good': 0, 'n_bad': 0, 'names': ['CV','Variable']}, }
    
    for modelID, model in master_data.items():
        
        true_class = model.model_class
        c = classes[true_class]
        
        if modelID in entry_data.keys():
            
            entry_model = entry_data[modelID]
            
            if entry_model.model_class in c['names']:
                c['n_good'] += 1
                log.info(' -> '+modelID+' ('+str(model.idx)+') correctly classified: true: '+true_class+\
                                    ' entry: '+entry_model.model_class)
                
            else:
                c['n_bad'] += 1
                log.info(' -> '+modelID+' ('+str(model.idx)+') misclassified: true: '+true_class+\
                                    ' entry: '+entry_model.model_class)
                                    
        else:
            
            log.info(' -> '+modelID+' ('+str(model.idx)+') No model found for event')
        
        classes[true_class] = c
    
    log.info('Number of events classified:')
    
    good_classes = []
    bad_classes = []
    axticks = []
    axlabels = []
    for i,key in enumerate(classes.keys()):
        good_classes.append( classes[key]['n_good'] )
        bad_classes.append( classes[key]['n_bad'] )
        axticks.append(i)
        axlabels.append(key)
        log.info(key+' '+str(classes[key]['n_good'])+' good, '+str(classes[key]['n_bad'])+' bad')
    axticks = np.array(axticks)
    
    fig = plt.figure(1,(10,10))
    
    bwidth = 0.4
    plt.subplot(1,1,1)

    p1 = plt.bar(axticks-bwidth/2.0, good_classes, 
                         bwidth, color='r', label='Accurately classified')

    
    p2 = plt.bar(axticks+bwidth/2.0, bad_classes, 
                         bwidth, color='k', label='Misclassified')
    
    plt.xlabel('Model type')
    plt.ylabel('Number classified')
    
    plt.xticks(axticks, axlabels)
    
    plt.grid(True)
    plt.legend()
    
    plt.savefig(path.join(params['log_dir'],'classifications.png'))
    


if __name__ == '__main__':
    
    evaluate_entry()
    