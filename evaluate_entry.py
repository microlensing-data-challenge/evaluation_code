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
    
    compare_parameters(params, master_data, entry_data, log)
    
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

def compare_class(true_class,entry_class):
    
    if true_class == 'PSPL' and entry_class == 'PSPL':
        
        return True
        
    if true_class == 'Binary_star' and entry_class == 'USBL':
        
        return True
        
    if true_class == 'Binary_planet' and entry_class == 'USBL':
        
        return True
    
    if true_class == 'CV' and entry_class in ['CV','Variable']:
        
        return True
    
    return False
    

def compare_parameters(params, master_data, entry_data, log):
    """Function to compare the entry's fitted parameters with those from the
    master table"""
    
    hdrs = ['ModelID', 'Class', 't0', 'tE', 'u0', 'rho', 'piE', \
            'fs<sub>W</sub>', 'fb<sub>W</sub>', 'fs<sub>Z</sub>', 'fb<sub>Z</sub>', \
            's', 'q', 'alpha', 'ds/dt', 'dalpha/dt', 'M1', 'M2', 'DL', 'DS', 'aperp',
            'chisq']
    par_list = ['t0', 'tE', 'u0', 'rho', 'piE', \
                'fs_W', 'fb_W', 'fs_Z', 'fb_Z', \
                's', 'q', 'alpha', 'dsdt', 'dadt', 'M1', 'M2', 'DL', 'DS', 'aperp']
    fpars = start_html_file(path.join(params['log_dir'],'parameters_evaluation.html'),hdrs)
    
    for modelID, model in master_data.items():
        
        true_class = model.model_class
    
        if modelID in entry_data.keys():
            
            entry_model = entry_data[modelID]
            
            if true_class in [ 'PSPL', 'Binary_star', 'Binary_planet']:
                
                line = '<tr><td>'+str(modelID)+'</td>'
                
                if compare_class(true_class,entry_model.model_class):
                    line = line + '<td>'+entry_model.model_class+'</td>'
                else:
                    line = line + '<td bgcolor="#F01F11">'+entry_model.model_class+'</td>'
                
                for par in par_list:
                    par_true = getattr(model,par)
                    par_fit = getattr(entry_model,par)
                    par_error = getattr(entry_model,'sig_'+par)
                    
                    (dpar,within_1sig,within_3sig) = compare_parameter(par_true,par_fit,par_error)
                    print(modelID, model.model_class, par,par_true,par_fit,par_error,within_1sig,within_3sig)
                    
                    if within_1sig and within_3sig:
                        line = line + ' <td> ' + str(par_fit)+' &plusmn; '+str(par_error) + '</td>'
                        
                    elif within_1sig == False and within_3sig:
                        line = line + ' <td bgcolor="#FFB623"> ' + str(par_fit)+' &plusmn; '+str(par_error)+'</td>'
                        
                    else:
                        line = line + ' <td bgcolor="#F01F11"> ' + str(par_fit)+' &plusmn; '+str(par_error)+'</td>'
                
                line = line + ' <td> '+str(entry_model.chisq_W)+' </td></tr>\n'
                
                fpars.write(line)
        else:
            
            if true_class in [ 'PSPL', 'Binary_star', 'Binary_planet']:
                
                line = str(modelID).replace('_','\_') + ' & - & - & - & - & - & - & - \\\\\n'

                fpars.write(line)
                
    fpars.write('</table>\n')
    fpars.write('</body>\n')
    fpars.write('</html>\n')
    fpars.close()

def compare_parameter(true_par,fitted_par,fitted_error):
    """Function to compare a fitted numerical parameter with the true model value"""
    
    within_1sig = False
    within_3sig = False
    
    if fitted_par == None and true_par == None:
        
        delta_par = None
        within_1sig = True
        within_3sig = True
    
    elif fitted_par != None and true_par != None:
        
        delta_par = true_par - fitted_par
        
        if fitted_error != None:
            one_sig = abs(fitted_par - fitted_error)
            three_sig = abs(fitted_par - (fitted_error * 3.0))
            
            if abs(delta_par) <= one_sig:
                within_1sig = True
                within_3sig = True
                
            if abs(delta_par) > one_sig and abs(delta_par) <= three_sig:
                within_3sig = True
    
    elif true_par == None and fitted_par != None:
        
        delta_par = 0.0
        
    elif true_par != None and fitted_par == None:
        
        delta_par = None
        
    return delta_par, within_1sig, within_3sig
    
def compare_times():
    """Function to evaluate the time taken to fit models"""
    
    pass

def start_html_file(file_path,header):
    """Function to start an HTML format table file"""
    
    f = open(file_path,'w')
    f.write('<html>\n')
    f.write('<body>\n')
    f.write('<table>\n')
    f.write('<tr>\n')
    l = ''
    for key in header:
        l = l + ('<th>'+key+'</th>')
    f.write(l+'<tr>\n')
    
    return f
    
def start_tex_file(file_path):
    """Function to start a latex format table file"""
    
    f = open(file_path,'w')
    f.write('\\documentclass[11pt]{article}\n')
    f.write('\\usepackage{amsmath,amssymb}\n')
    f.write('\\usepackage{xcolor,colortbl}\n')
    f.write('\\usepackage{float}\n')
    f.write('\\begin{document}\n')
    
    f.write('\\begin{table}\n')
    f.write('\\centering\n')
    f.write('\\begin{tabular}{llllllll}\n')
    f.write('\\hline\n')
    
    return f
   
if __name__ == '__main__':
    
    evaluate_entry()
    