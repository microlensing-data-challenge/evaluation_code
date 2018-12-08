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
    
    
    categories = { 'PSPL': ['PSPL','single-lens'],
                   'Binary_star': ['USBL','binary-lens'],
                   'Binary_planet': ['USBL','binary-lens', 'planet'],
                   'CV': ['CV','Variable','variable'] }
                
    params = get_args()
    
    log = start_log( params['log_file'] )
    
    summary_path = path.join(params['log_dir'],'entry_summary.html')
    
    summary = start_html_file(summary_path,title=params['teamID'])
    
    master_data = parse_table1.read_master_table(params['master_file'])
    
    entry_data = parse_table1.read_standard_ascii_DC_table(params['entry_file'])
    
    summary = check_classifications(params, master_data, entry_data, 
                                    categories, summary, log)
    
    (deltas,summary) = compare_parameters(params, master_data, entry_data, 
                                    categories, summary, log)
    
    summary = plot_deltas(params,deltas,summary, log)
    
    summary = compare_times(params,entry_data,categories,summary,log)
    
    log.info( 'Analysis complete\n' )
    logging.shutdown()
    
    summary.write('</body>\n')
    summary.write('</html>\n')
    summary.close()
    
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
        params['teamID'] = input('Please enter the path to the entry data file: ')
        
    else:
        
        params['master_file'] = argv[1]
        params['entry_file'] = argv[2]
        params['teamID'] = argv[3]
    
    params['log_dir'] = path.dirname(params['entry_file'])
    params['log_file'] = path.join(params['log_dir'], 'evaluation.log')
        
    return params

def check_classifications(params, master_data, entry_data, categories, summary, log):
    """Function to check whether models have been correctly classified"""
    
    log.info('Checking model classifications')
    
    # For each class, record [ n_good, n_bad ] classifications
    classes = { 'PSPL': {'n_good': 0, 'n_bad': 0, 'names': categories['PSPL']},
                'Binary_star': {'n_good': 0, 'n_bad': 0, 'names': categories['Binary_star']},
                'Binary_planet': {'n_good': 0, 'n_bad': 0, 'names': categories['Binary_planet']},
                'CV': {'n_good': 0, 'n_bad': 0, 'names': categories['CV']} }
    
    true_classes = list_classes(master_data)
    fitted_classes = list_classes(entry_data)
    confuse_matrix = np.zeros([len(true_classes),len(fitted_classes)])
    
    for modelID, model in master_data.items():
        
        true_class = model.model_class
        c = classes[true_class]
        
        if modelID in entry_data.keys():
            
            entry_models = entry_data[modelID]
            
            for m in entry_models:
                confuse_matrix = record_confusion_matrix(true_class,m.model_class,
                                                         confuse_matrix,
                                                         true_classes,fitted_classes)
                
                if m.model_class in c['names']:
                    c['n_good'] += 1
                    log.info(' -> '+modelID+' ('+str(model.idx)+') correctly classified: true: '+true_class+\
                                        ' entry: '+m.model_class)
                    
                else:
                    c['n_bad'] += 1
                    log.info(' -> '+modelID+' ('+str(model.idx)+') misclassified: true: '+true_class+\
                                        ' entry: '+m.model_class)
                                    
        else:
            
            log.info(' -> '+modelID+' ('+str(model.idx)+') No model found for event')
        
        classes[true_class] = c
    
    log.info('Number of classifications:')
    
    summary.write('<br><h2>Classification</h2>\n')
    summary.write('<p>Note that these totals include the classifications from all fitted models.  If more than one fitted model was provided for a simulated event, they may exceed the total number of events.\n')
    summary.write('<table>\n')
    
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
        summary.write('<tr><td>'+key+'</td><td>'+str(classes[key]['n_good'])+' good</td><td>'+str(classes[key]['n_bad'])+' bad</td></tr>')
    axticks = np.array(axticks)
    
    summary.write('</table><br>\n')
    
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
    
    plt.savefig(path.join(params['log_dir'],'classifications.png'),bbox_inches='tight')
    
    plt.close(1)

    fig = plt.figure(2,(10,10))
    
    plt.imshow(confuse_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    
    plt.colorbar()
    
    xtick_marks = np.arange(len(fitted_classes))
    plt.xticks(xtick_marks, fitted_classes, rotation=45)
    ytick_marks = np.arange(len(true_classes))
    plt.yticks(ytick_marks, true_classes)

    plt.ylabel('True class')
    plt.xlabel('Fitted class')
    
    plt.savefig(path.join(params['log_dir'],'confusion_matrix.png'),bbox_inches='tight')
    
    plt.close(2)
    
    summary.write('<table>\n')
    summary.write('<tr><td><img src="classifications.png" width="100%"></td><td><img src="confusion_matrix.png"  width="100%"></td></tr>\n')
    summary.write('</table><br>\n')
    
    return summary
    
def list_classes(event_list):
    
    class_list = []
    
    for entryID, entry in event_list.items():
        
        if type(entry) == type([]):
            
            for m in entry:
                
                if m.model_class not in class_list:
                    
                    class_list.append(m.model_class)
            
        else:
            
            if entry.model_class not in class_list:
                    
                class_list.append(entry.model_class)
                
    return class_list
    
def compare_class(true_class,entry_class,categories):
    
    if true_class == 'PSPL' and entry_class in categories['PSPL']:
        
        return True
        
    if true_class == 'Binary_star' and entry_class in categories['Binary_star']:
        
        return True
        
    if true_class == 'Binary_planet' and entry_class in categories['Binary_planet']:
        
        return True
    
    if true_class == 'CV' and entry_class in categories['CV']:
        
        return True
    
    return False
    
def record_confusion_matrix(true_class,fitted_class,confuse_matrix,
                            true_classes,fitted_classes):
    """Function to log which objects are identified as which classes"""
    
    itrue = true_classes.index(true_class)
    ifit = fitted_classes.index(fitted_class)
    
    confuse_matrix[itrue,ifit] += 1
    
    return confuse_matrix
    
def compare_parameters(params, master_data, entry_data, categories,
                       summary, log, colour_coding=False):
    """Function to compare the entry's fitted parameters with those from the
    master table"""
    
    hdrs = ['ModelID', 'Class', 't0', 'tE', 'u0', 'rho', 'piE', \
            'fs<sub>W</sub>', 'fb<sub>W</sub>', 'fs<sub>Z</sub>', 'fb<sub>Z</sub>', \
            's', 'q', 'alpha', 'ds/dt', 'dalpha/dt', 'M1', 'M2', 'DL', 'DS', 'aperp',
            'chisq']
    par_list = ['t0', 'tE', 'u0', 'rho', 'piE', \
                'fs_W', 'fb_W', 'fs_Z', 'fb_Z', \
                's', 'q', 'alpha', 'dsdt', 'dadt', 'M1', 'M2', 'DL', 'DS', 'aperp']
    fpars = start_html_file(path.join(params['log_dir'],'parameters_evaluation.html'))
    fpars.write('<center><h1>Comparison of fitted parameters for '+params['teamID']+'</h1></center>\n')
    fpars.write('<br>\n')
    fpars.write('<p>The table below compares the parameters obtained during the fitting process (black) with the true parameters used to simulate the datasets (grey, italics)</p>\n')
    fpars.write('<p>If a team provided several alternative models for a single dataset, these are represented by multiple entries with the same ModelID</p>\n')
    fpars.write("<p>'None' entries represent values missing from the team's table entry data.</p>\n")
    fpars.write('<p><a href="entry_summary.html">Back to entry summary</a>\n')
    
    fpars = start_html_table(fpars,hdrs)

    priority_pars = ['t0', 'tE', 'u0', 'piE', 'fs_W', 'fb_W', 'fs_Z', 'fb_Z', 's', 'q', 'alpha']
    
    deltas = { 'PSPL_true': {}, 'PSPL_false': {},
               'Binary_star_true': {}, 'Binary_star_false': {},
               'Binary_planet_true': {}, 'Binary_planet_false': {},
              }
    for group in deltas.keys():
        for key in priority_pars:
            deltas[group][key] = []
    
    log.info('Comparing parameter fitted values with those simulated')
    
    for modelID, model in master_data.items():
        
        true_class = model.model_class
    
        if modelID in entry_data.keys():
            
            entry_models = entry_data[modelID]
            
            for m in entry_models:
                if true_class in [ 'PSPL', 'Binary_star', 'Binary_planet']:
                    
                    line = '<tr><td>'+str(modelID)+'</td>'
                    
                    accurate_class = compare_class(true_class,m.model_class,categories)
                    group = get_model_group(true_class,accurate_class)
                    
                    if accurate_class:
                        line = line + '<td>'+m.model_class+'<br><font color="#515A5A"><i>'+model.model_class+'</i></font></td>'
                    elif accurate_class == False and colour_coding:
                        line = line + '<td bgcolor="#F08080">'+m.model_class+'<br><font color="#515A5A"><i>'+model.model_class+'</i></font></td>'
                    else:
                        line = line + '<td>'+m.model_class+'<br><font color="#515A5A"><i>'+model.model_class+'</i></font></td>'
                    
                    for par in par_list:
                        par_true = getattr(model,par)
                        par_fit = getattr(m,par)
                        par_error = getattr(m,'sig_'+par)
                        
                        (dpar,within_1sig,within_3sig) = compare_parameter(par_true,par_fit,par_error)
                        
                        #if par == 't0':   
                         #   print(modelID, model.model_class, par,par_true,par_fit,dpar,par_error,within_1sig,within_3sig)
                        
                        if colour_coding:
                            if within_1sig and within_3sig:
                                line = line + ' <td> ' + str(par_fit)+' &plusmn; '+str(par_error) + \
                                        '<br><font color="#515A5A"><i>'+ str(par_true)+'</i></font></td>'
                                
                            elif within_1sig == False and within_3sig:
                                line = line + ' <td bgcolor="#F7DC6F"> ' + str(par_fit)+' &plusmn; '+str(par_error)+\
                                        '<br><font color="#515A5A"><i>'+ str(par_true)+'</i></font></td>'
                                
                            else:
                                line = line + ' <td bgcolor="#F08080"> ' + str(par_fit)+' &plusmn; '+str(par_error)+\
                                        '<br><font color="#515A5A"><i>'+ str(par_true)+'</i></font></td>'
                        else:
                            line = line + ' <td> ' + str(par_fit)+' &plusmn; '+str(par_error) + \
                                        '<br><font color="#515A5A"><i>'+ str(par_true)+'</i></font></td>'
                                        
                        if dpar != None and par in priority_pars:
                            deltas[group][par].append(dpar)
                                
                    line = line + ' <td> '+str(m.chisq_W)+' </td></tr>\n'
                    
                    fpars.write(line)
        else:
            
            if true_class in [ 'PSPL', 'Binary_star', 'Binary_planet']:
                
                line = str(modelID).replace('_','\_') + ' & - & - & - & - & - & - & - \\\\\n'

                fpars.write(line)
                
    fpars.write('</table>\n')
    fpars.write('</body>\n')
    fpars.write('</html>\n')
    fpars.close()
    
    summary.write('<br><h2>Comparison of parameters with simulated values</h2>\n')
    summary.write('<p><a href="parameters_evaluation.html">Cross-matched parameter table</a></p>\n')
    
    return deltas, summary

def get_model_group(true_class,accurate_class):
    
    if accurate_class:
        group = true_class+'_true'
    else:
        group = true_class+'_false'
    
    return group
    
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
            
            if abs(delta_par) <= fitted_error:
                within_1sig = True
                within_3sig = True
                
            if abs(delta_par) > fitted_error and abs(delta_par) <= (3.0*fitted_error):
                within_3sig = True
    
    elif true_par == None and fitted_par != None:
        
        delta_par = None
        
    elif true_par != None and fitted_par == None:
        
        delta_par = None
        
    return delta_par, within_1sig, within_3sig

def plot_deltas(params,deltas,summary, log):
    """Function to plot distributions of the differences between the fitted 
    and true parameters"""

    priority_pars = ['t0', 'tE', 'u0', 'piE','fs_W', 'fb_W', 'fs_Z', 'fb_Z', 's', 'q', 'alpha']
    headers = ['Parameter', 'Mean diff', 'Median diff', 'St. Dev', 'Min diff', 'Max diff', 'N models fitted']
    
    # xmin, xmax, nbins
    plot_limits = {'t0': [-20.0, 20.0, 100],
                'tE': [-20.0, 20.0, 100],
                'u0': [-2.0, 2.0, 100],
                'piE': [-5.0, 5.0, 100],
                'fs_W': [-1e4, 1e4, 200],
                'fb_W': [-1e4, 1e4, 200],
                'fs_Z': [-1e4, 1e4, 200],
                'fb_Z': [-1e4, 1e4, 200],
                's': [-5.0, 5.0, 100],
                'q': [-5.0, 5.0, 100],
                'alpha': [-5.0, 5.0, 100],
                }
    group_colours = {'PSPL_true': '#05C709', 'PSPL_false': '#026604', 
                     'Binary_star_true': '#5D5FFD', 'Binary_star_false': '#020499',
                     'Binary_planet_true': '#CC03F5', 'Binary_planet_false': '#7F0299'}
                     
    log.info('Plotting distributions between fitted and true parameters')
    log.info('\n Parameter mean_diff, median_diff, St.Dev min  max N_models')
    
    summary.write('<p><h2>Distributions of fitted parameters from true values</h2>\n')
    summary.write('<p>The tables and plots below provide basic statistics on the distributions of fitted parameter values relative to the true simulated data, with separate analysis performed for correctly and incorrectly classified models.</p>\n')
    summary.write('<p>Please note that parameter values are frequently not provided for models which were missclassified.  No comparison can be made in these cases.</p>\n')
    
    for group in deltas.keys():
        summary.write('<p><b>'+group.replace('_',' ')+'</b><tr>\n')
        log.info(group)
        summary = start_html_table(summary,headers)
        
        for par,values in deltas[group].items():
            
            if len(values) > 0:
                data = np.array(values)
            
                median = np.median(data)
                
                log.info(par+' '+str(data.mean())+' '+str(median)+' '+str(data.std())+' '+str(data.min())+' '+str(data.max())+' '+str(len(data)))

                summary.write('<tr><td>'+par+'</td><td>'+str(data.mean())+'</td><td>'+str(median)+'</td><td>'+str(data.std())+'</td><td>'+str(data.min())+'</td><td>'+str(data.max())+'</td><td>'+str(len(data))+'</td></tr>\n')
            
            else:
                log.info(par+' No models fitted with this parameter')
                
                summary.write('<tr><td>'+par+'</td><td><td colspan="6">No models fitted with this parameter</td></tr>')
            
        summary.write('</table>\n')
        summary.write('</p><br>\n')
    
    plt_dict = {'PSPL': {}, 'Binary_star': {}, 'Binary_planet': {}}
    
    for par in deltas['Binary_star_true'].keys():
        
        limits = plot_limits[par]
        
        for group in plt_dict.keys():
            
            values_true = deltas[group+'_true'][par]
            values_false = deltas[group+'_false'][par]
            
            if len(values_true) > 0 or len(values_false) > 0:
                
                fig = plt.figure(1,(10,10))
                
                plt.subplot(1,1,1)
            
                data_true = np.array(values_true)
                data_false = np.array(values_false)
                
                (n, bins, patches) = plt.hist(data_true, limits[2], 
                                                 facecolor='#026604', 
                                                  alpha=0.75,label='Classified')
                                                  
                (n, bins, patches) = plt.hist(data_false, bins, 
                                                  facecolor='#7F0299', 
                                                  alpha=0.75,label='Missclassified')
                
                plt.title('Distribution in $\delta '+par+'$', fontsize=18)
                plt.xlabel('$\delta '+par+'$', fontsize=18)
                plt.ylabel('Frequency', fontsize=18)
                
                (xmin,xmax,ymin,ymax) = plt.axis()
                (xmin,xmax) = get_xlimits(data_true,data_false)
                plt.axis([xmin,xmax,ymin,ymax])
                #plt.axis([limits[0],limits[1],ymin,ymax])
                
                plt.grid(True)
                plt.legend()
                
                plt.tick_params(axis='x', labelsize=18)
                plt.tick_params(axis='y', labelsize=18)
        
                plt.savefig(path.join(params['log_dir'],'delta_'+group+'_'+par+'_distribution.png'), bbox_inches='tight')
        
                plt.close(1)
            
                plt_dict[group][par] = True
                
            else:
                plt_dict[group][par] = False
    
    summary.write('<table>\n<tr>')
    header = '<tr><th><strong>Parameter</strong></th>'
    for k in plt_dict.keys():
        header += '<th>'+k+'</th>'
    header += '</tr>'
    summary.write(header+'\n')
    
    for par in deltas['Binary_star_true'].keys():
        
        summary.write('<tr><td><strong>'+par+'</strong></td>')

        for group in ['PSPL', 'Binary_star', 'Binary_planet']:
            
            if plt_dict[group][par]:
                summary.write('<td><img src="delta_'+group+'_'+par+'_distribution.png" width="100%"></td>')
            else:
                summary.write('<td>No data</td>')
        summary.write('</tr>\n')
    summary.write('</table>\n')
    summary.write('<br>\n')
    
    return summary

def get_xlimits(data_true,data_false):
    
    if len(data_true) > 0 and len(data_false) > 0:
        xmin = min(data_true.min(), data_false.min())
        xmax = max(data_true.max(), data_false.max())
        
    elif len(data_true) > 0 and len(data_false) == 0:
        xmin = data_true.min()
        xmax = data_true.max()
        
    elif len(data_true) == 0 and len(data_false) > 0:
        xmin = data_false.min()
        xmax = data_false.max()
    
    return xmin, xmax
    
def compare_times(params,entry_data,categories,summary,log):
    """Function to evaluate the time taken to fit models"""
    
    log.info('Plotting distribution of time taken for fit')
    
    
    ts_pspl = []
    ts_binary = []
    
    for modelID, model_list in entry_data.items():
        
        for m in model_list:
            if m.t_fit != None and m.model_class in categories['PSPL']:
                ts_pspl.append(m.t_fit)
                
            if m.t_fit != None and m.model_class in categories['Binary_star']:
                ts_binary.append(m.t_fit)
    
    ts_pspl = np.array(ts_pspl)
    ts_binary = np.array(ts_binary)
    
    fig = plt.figure(1,(20,10))
        
    plt.subplot(1,2,1)
    
    if len(ts_pspl) > 0:
        (n, bins, patches) = plt.hist(ts_pspl, 50, facecolor='g', alpha=0.75,
                                label='PSPL fits')
        
    plt.title('PSPL fits')
    plt.xlabel('Time to fit [hrs]')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    if len(ts_binary) > 0:
        plt.subplot(1,2,2)
        
        if len(ts_binary) > 0:
            (n, bins, patches) = plt.hist(ts_binary, 50, facecolor='m', alpha=0.75,
                                    label='Binary fits')
        
        plt.title('Binary fits')
        plt.xlabel('Time to fit [hrs]')
        plt.ylabel('Frequency')
        plt.grid(True)
    
    plt.savefig(path.join(params['log_dir'],'time_to_fit_distribution.png'), bbox_inches='tight')

    plt.close(1)

    summary.write('<br><h2>Time to compute fits</h2>\n')
    summary.write('<br>\n')    
    
    if len(ts_pspl) > 0:
        median_pspl = np.median(ts_pspl)
        log.info('Median time taken for PSPL fit = '+str(median_pspl)+\
                                    ' std.dev. = '+str(ts_pspl.std()))
        summary.write('Median time taken for PSPL fit = '+str(median_pspl)+\
                                    ' std.dev. = '+str(ts_pspl.std())+'\n')
    else:
        log.info('Fitting times not recorded for PSPL models')
        summary.write('Fitting times not recorded for PSPL models\n')
        
    if len(ts_binary) > 0:
        median_binary = np.median(ts_binary)
        log.info('Median time taken for binary fit = '+str(median_binary)+\
                                    ' std.dev. = '+str(ts_binary.std()))
        summary.write('Median time taken for binary fit = '+str(median_binary)+\
                                    ' std.dev. = '+str(ts_binary.std())+'\n')
    else:
        log.info('Fitting times not recorded for binary models')
        summary.write('Fitting times not recorded for binary models\n')
    
    summary.write('<img src="time_to_fit_distribution.png" width="100%">\n')
    
    return summary
    
def start_html_table(htmlfile,header):
    """Function to start an HTML format table file in an open file object"""
    
    htmlfile.write('<html>\n')
    htmlfile.write('<body>\n')
    htmlfile.write('<table>\n')
    htmlfile.write('<tr>\n')
    l = ''
    for key in header:
        l = l + ('<th>'+key+'</th>')
    htmlfile.write(l+'<tr>\n')
    
    return htmlfile

def start_html_file(file_path,title=None):
    """Function to start an HTML format table file"""
    
    f = open(file_path,'w')
    f.write('<html>\n')
    f.write('<body>\n')
    if title != None:
        f.write('<center><h1>'+title+'</h1></center>\n')
   
    return f
   
if __name__ == '__main__':
    
    evaluate_entry()
        