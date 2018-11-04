# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:30:07 2018

@author: rstreet
"""

from os import path
from sys import argv
import numpy as np
import copy

class EventEntry():
    """Class describing the attributes of the parameters for a given event 
    in the data table provided for a Data Challenge entry
    """
    
    def __init__(self, kwargs=None):
        
        self.idx = None
        self.modelID = None
        self.model_class = None
        self.t0 = None
        self.sig_t0 = None
        self.tE = None
        self.sig_tE = None
        self.u0 = None
        self.sig_u0 = None
        self.rho = None
        self.sig_rho = None
        self.piEE = None
        self.sig_piEE = None
        self.piEN = None
        self.sig_piEN = None
        self.fs1 = None
        self.sig_fs1 = None
        self.fb1 = None
        self.sig_fb1 = None
        self.fs2 = None
        self.sig_fs2 = None
        self.fb2 = None
        self.sig_fb2 = None
        self.s = None
        self.sig_s = None
        self.q = None
        self.sig_q = None
        self.alpha = None
        self.sig_alpha = None
        self.dsdt = None
        self.sig_dsdt = None
        self.dadt = None
        self.sig_dadt = None
        self.t0_par = None
        self.chisq1  = None
        self.chisq2  = None
        self.M1 = None
        self.sig_M1 = None
        self.M2 = None
        self.sig_M2 = None
        self.DL = None
        self.sig_DL = None
        self.DS = None
        self.sig_DS = None
        self.aperp = None
        self.sig_aperp = None
        self.t_fit = None

        self.requirements = [
                    ('idx','int',None, 'required'),
                    ('modelID','str',None, 'required'),
                    ('model_class', 'str', None, 'required'),
                    ('t0', 'float', [2450000.0,2470000.0], 'required'),
                    ('sig_t0', 'float', [1000.0,10000.0], 'required'), 
                    ('tE', 'float', [0.0,500.0], 'required'),
                    ('sig_tE', 'float', [0.0,500.0], 'required'),
                    ('u0', 'float', [-5.0,5.0], 'required'),
                    ('sig_u0', 'float', [-5.0,5.0], 'required'),
                    ('rho', 'float', [0.0,1.0], None),
                    ('sig_rho', 'float', [0.0,1.0], None),
                    ('piEE','float', [-5.0,5.0], None), 
                    ('sig_piEE', 'float', [-5.0,5.0], None),
                    ('piEN', 'float', [-5.0,5.0], None),
                    ('sig_piEN', 'float', [-5.0,5.0], None),
                    ('fs1', 'float', [-500.0,500000.0], 'required'),
                    ('sig_fs1', 'float', [0.0,50000.0], 'required'),
                    ('fb1', 'float', [-500.0,50000.0], 'required'),
                    ('sig_fb1', 'float', [0.0,50000.0], 'required'),
                    ('fs2', 'float', [-500.0,500000.0], 'required'),
                    ('sig_fs2', 'float', [0.0,50000.0], 'required'),
                    ('fb2', 'float', [-500.0,50000.0], 'required'),
                    ('sig_fb2', 'float', [0.0,50000.0], 'required'),
                    ('s', 'float', [0.0,50.0], 'required'),
                    ('sig_s', 'float', [0.0,50.0], 'required'),
                    ('q', 'float', [0.0,1.0], 'required'),
                    ('sig_q', 'float', [0.0,1.0], 'required'),
                    ('alpha', 'float', [0.0,6.4], 'required'),
                    ('sig_alpha', 'float', [0.0,6.4], 'required'), 
                    ('dsdt', 'float', [0.0,50.0], None),
                    ('sig_dsdt', 'float', [0.0,50.0], None), 
                    ('dadt', 'float', [0.0,6.4], None), 
                    ('sig_dadt', 'float', [0.0,6.4], None),
                    ('t0_par', 'float', [2450000.0,2470000.0], None), 
                    ('chisq1', 'float', [0.0,1000000.0], 'required'), 
                    ('chisq2', 'float', [0.0,1000000.0], 'required'), 
                    ('M1', 'float', [0.0,20.0], None),
                    ('sig_M1', 'float', [0.0,20.0], None),
                    ('M2', 'float', [0.0,20.0], None),
                    ('sig_M2', 'float', [0.0,20.0], None),
                    ('DL', 'float', [0.0,10000.0], None),
                    ('sig_DL', 'float', [0.0,10000.0], None),
                    ('DS', 'float', [0.0,10000.0], None),
                    ('sig_DS', 'float', [0.0,10000.0], None),
                    ('aperp', 'float', [0.0,100.0], None),
                    ('sig_aperp', 'float', [0.0,100.0], None),
                    ('t_fit', 'float', [0.0,3600.0], 'required'),
        ]        
        
        if kwargs != None:
            
            for key, value in kwargs.items():
                
                setattr(self,key,value)
    
    def self_check(self):
        
        for pars in self.requirements:
            
            (key, form, allowed_range, required) = pars
            
            got_value = False
            format_ok = False
            within_range = False
            got_error = False
            
            if required:
                
                item = getattr(self,key)
                
                if item == None:
                    
                    print('Error: No entry for '+key+' for entry '+\
                            str(self.idx)+', '+repr(self.modelID))
                
                else:
                    
                    got_value = True
                    
                if form == 'int' and type(item) != type(0):
                    
                    print('Error: Entry for '+key+\
                            ' should be an integer, got '+repr(type(item)))
                    
                elif form == 'float' and type(item) != type(0.0):
                    
                    print('Error: Entry for '+key+\
                            ' should be a float, got '+repr(type(item)))
                    
                elif form == 'str' and type(item) != type('test'):
                    
                    print('Error: Entry for '+key+\
                            ' should be a string, got '+repr(type(item)))
                    
                else:
                    
                    format_ok = True
                    
                if allowed_range != None and got_value:
                    
                    if item >= allowed_range[0] and item <= allowed_range[1]:
                        
                        within_range = True
                    
                else:
                    
                    within_range = True
                    
            
            if 'sig' not in key and got_value and \
                key not in ['idx', 'modelID', 'model_class', 'chisq']:
                
                sig_key = 'sig_'+key
                
                try:
                    sig_value = getattr(self,sig_key)
                    
                    if sig_value != None:
                        
                        got_error = True
                
                except AttributeError:
                    
                    sig_value = None
                    
    def summary(self):
        
        output = ''
        for par in self.requirements:
            key = par[0]
            
            output += ' '+key+'='+repr(getattr(self,key))
            
        return output
        
def read_standard_ascii_DC_table(file_path):
    """Function to read a Data Challenge entry table in standard ASCII format
    
    Expected format:
    # Column 1: Model ID containing both target ID and solution number e.g. [target]_[solution]<br>
    # Column 2: Classification<br>
    # Column 3: Time of peak, t0 [days] - priority<br>
    # Column 4: Uncertainty in t0 [days] - priority<br>
    # Column 5: Einstein crossing time, tE [days] - priority<br>
    # Column 6: Uncertainty in tE [days] - priority<br>
    # Column 7: Minimum impact parameter, u0 [normalised by θE] - priority<br>
    # Column 8: Uncertainty in u0 - priority<br>
    # Column 9: Angular source size parameter, rho<br>
    # Column 10: Uncertainty on rho<br>
    # Column 11: Parallax parameter πE,E<br>
    # Column 12: Uncertainty on πE,E<br>
    # Column 13: Parallax parameter πE,N<br>
    # Column 14: Uncertainty on πE,N<br>
    # Column 15: Source flux, fs, filter 1 [counts/s] - priority<br>
    # Column 16: Uncertainty in fs, filter 1 [counts/s] - priority<br>
    # Column 17: Blend flux, fb, filter 1 [counts/s] - priority<br>
    # Column 18: Uncertainty in fb, filter 1 [counts/s] - priority<br>
    # Column 19: Source flux, fs, filter 2 [counts/s] - priority<br>
    # Column 20: Uncertainty in fs, filter 2 [counts/s] - priority<br>
    # Column 21: Blend flux, fb, filter 2 [counts/s] - priority<br>
    # Column 22: Uncertainty in fb, filter 2 [counts/s] - priority<br>
    # Column 23: Binary separation, s, [normalised by θE] - priority<br>
    # Column 24: Uncertainty on s - priority<br>
    # Column 25: Mass ratio, q = M2/M1 - priority<br>
    # Column 26: Uncertainty on q - priority<br>
    # Column 27: Angle of lens motion, alpha - priority<br>
    # Column 28: Uncertainty on alpha - priority<br>
    # Column 29: Rate of change of s, ds/dt<br>
    # Column 30: Uncertainty on ds/dt<br>
    # Column 31: Rate of change of alpha, dalpha/dt<br>
    # Column 32: Uncertainty on dalpha/dt<br>
    # Column 33: t0_par [days]<br>
    # Column 34: Chi squared of the fitted model, filter 1<br>
    # Column 34: Chi squared of the fitted model, filter 2<br>
    # Column 35: Primary lens mass, M1 [Msolar]<br>
    # Column 36: Uncertainty on M1 [Msolar]<br>
    # Column 37: Secondary lens mass, M2 [MJupiter]<br>
    # Column 38: Uncertainty on M2 [MJupiter]<br>
    # Column 39: Distance to the lens, DL [kpc]<br>
    # Column 40: Uncertainty on DL [kpc]<br>
    # Column 41: Distance to the source, DS [kpc]<br>
    # Column 42: Uncertainty on DS [kpc]<br>
    # Column 43: Projected separation of lens masses, aperp [AU]<br>
    # Column 44: Uncertainty on aperp [AU]<br>
    # Column 45: Time taken to fit the lightcurve from data ingest to final output [hrs]<br>
    """
    
    if path.isfile(file_path) == False:
        
        print('Error: Cannot find file '+file_path)
        
        exit()
    
    file_lines = open(file_path,'r').readlines()
    
    event_list = []
    data = []
    header = []
    
    for i,line in enumerate(file_lines):
        
        if line[0:1] != '#':
            
            items = line.replace('\n','').split()
            
            if len(header) == 0:
                
                header = items[2:]
                
            else:
                
                entry = EventEntry({'idx': (len(event_list)+1), 
                                    'modelID': items[0], 
                                    'model_class': items[1]})
        
                values = []
                
                for j,f in enumerate(items[2:]):
                    
                    print(j,f )
                    
                    if 'None' in str(f):
                        
                        values.append(np.nan)
                        
                    else:
                        
                        values.append(float(f))
                        
                        setattr(entry,header[j],float(f))
                        
                entry.self_check()
                
                if len(values) == 39:
                    
                    event_list.append(entry)
                    data.append(values)
                
                else:
                    
                    print('Error parsing line '+str(i)+' in '+path.basename(file_path))
                    
    data = np.array(data)
    
    return event_list, data

def read_master_table(file_path):
    """Function to read the input file of the original simulation parameters
    per event"""
    
    if path.isfile(file_path) == False:
        raise IOError('Cannot find input file '+file_path)
        exit()
        
    lines = open(file_path,'r').readlines()
    
    master_data = []
    
    for i,l in enumerate(lines):
        
        if '#' not in l:
            entries = l.replace('\n','').split()
            
            e = EventEntry()
            
            if 'dcnormffp' in l: # PSPL, inc FFP
                pars = { 'model_class': 'PSPL',
                         'idx': 95,
                        }
                        
            elif 'ombin' in l: # Binary star
                pars = { 'model_class': 'Binary_star',
                         'idx': 81,
                         }
            
            elif 'omcassan' in l:   # Bound planet
                 pars = { 'model_class': 'Binary_planet',
                         'idx': 81,
                         }
            
            if 'dccv' in l: # CV
                 pars = { 'model_class': 'CV',
                         'idx': 79,
                         }
            
            e.t0 = float(entries[32])
            e.sig_t0 = 0.0
            e.tE = float(entries[33])
            e.sig_tE = 0.0
            e.u0 = float(entries[30])
            e.sig_u0 = 0.0
            e.rho = float(entries[37])
            e.sig_rho = 0.0
            e.piEE = float(entries[36])     # piE -> piEN, piEE
            e.sig_piEE = 0.0
            e.piEN = None
            e.sig_piEN = None
            e.fs1 = float(entries[66])       # -> 67 for second filter
            e.sig_fs1 = 0.0
            e.fb1 = None
            e.sig_fb1 = None
            e.fs2 = float(entries[67])       # -> 67 for second filter
            e.sig_fs2 = 0.0
            e.fb2 = None
            e.sig_fb2 = None
            e.s = float(entries[47])
            e.sig_s = 0.0
            e.q = float(entries[46])
            e.sig_q = 0.0
            e.alpha = float(entries[31])
            e.sig_alpha = 0.0
            e.dsdt = None                   # Can we derive these from the fit?
            e.sig_dsdt = None
            e.dadt = None
            e.sig_dadt = None
            e.t0_par = None
            e.chisq1  = float(entries[73])
            e.chisq2  = float(entries[74])
            e.M1 = float(entries[19])       # M_total or M1?
            e.sig_M1 = None
            e.M2 = float(entries[42])
            e.sig_M2 = None
            e.DL = float(entries[18])
            e.sig_DL = None
            e.DS = float(entries[9])
            e.sig_DS = None
            e.aperp = float(entries[43])    # a not aperp
            e.sig_aperp = None
            e.t_fit = 0.0
            
            for key, icol in pars.items():
    
                if key in ['model_class']:
                    setattr(e, key, icol)
                elif key in ['idx']:
                    setattr(e, key, int(entries[icol]))
                else:
                    setattr(e, key, float(entries[icol]))
                
                if key == 'idx':
                    setattr(e, 'modelID', 'ulwcd1_'+str(entries[icol]))
            
            master_data.append(e)
            
            print(e.summary())
            
            cont = input('Continue? ')
            
    return master_data
    
if __name__ == '__main__':
    
    if len(argv) == 1:
        
        file_path = input('Please enter the path to the table file: ')
        opt = input('Is this a master table? Y or n: ')
        
    else:
        
        file_path = argv[1]
        opt = argv[2]

    if 'Y' in opt or 'master' in opt:
        event_list = read_master_table(file_path)
    else:
        (event_list, data) = read_standard_ascii_DC_table(file_path)
    
    for i in range(0,len(event_list),1):
        
        print(event_list[i].summary())
    