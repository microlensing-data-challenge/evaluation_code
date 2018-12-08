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
        self.true_class = None
        self.t0 = None
        self.sig_t0 = None
        self.tE = None
        self.sig_tE = None
        self.u0 = None
        self.sig_u0 = None
        self.rho = None
        self.sig_rho = None
        self.piE = None
        self.sig_piE = None
        self.piEE = None
        self.sig_piEE = None
        self.piEN = None
        self.sig_piEN = None
        self.fs_W = None
        self.sig_fs_W = None
        self.fb_W = None
        self.sig_fb_W = None
        self.fs_Z = None
        self.sig_fs_Z = None
        self.fb_Z = None
        self.sig_fb_Z = None
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
        self.chisq_W  = None
        self.chisq_Z = None
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
                    ('piE','float', [-5.0,5.0], None), 
                    ('sig_piE', 'float', [-5.0,5.0], None),
                    ('piEE','float', [-5.0,5.0], None), 
                    ('sig_piEE', 'float', [-5.0,5.0], None),
                    ('piEN', 'float', [-5.0,5.0], None),
                    ('sig_piEN', 'float', [-5.0,5.0], None),
                    ('fs_W', 'float', [-500.0,500000.0], 'required'),
                    ('sig_fs_W', 'float', [0.0,50000.0], 'required'),
                    ('fb_W', 'float', [-500.0,50000.0], 'required'),
                    ('sig_fb_W', 'float', [0.0,50000.0], 'required'),
                    ('fs_Z', 'float', [-500.0,500000.0], 'required'),
                    ('sig_fs_Z', 'float', [0.0,50000.0], 'required'),
                    ('fb_Z', 'float', [-500.0,50000.0], 'required'),
                    ('sig_fb_Z', 'float', [0.0,50000.0], 'required'),
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
                    ('chisq_W', 'float', [0.0,1000000.0], 'required'), 
                    ('chisq_Z', 'float', [0.0,1000000.0], 'required'), 
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
    
def read_standard_ascii_DC_table(file_path,page=False):
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
    # Column 15: Source flux, fs, filter W149 [counts/s] - priority<br>
    # Column 16: Uncertainty in fs, filter W194 [counts/s] - priority<br>
    # Column 17: Blend flux, fb, filter W149 [counts/s] - priority<br>
    # Column 18: Uncertainty in fb, filter W149 [counts/s] - priority<br>
    # Column 19: Source flux, fs, filter Z087 [counts/s] - priority<br>
    # Column 20: Uncertainty in fs, filter Z087 [counts/s] - priority<br>
    # Column 21: Blend flux, fb, filter Z087 [counts/s] - priority<br>
    # Column 22: Uncertainty in fb, filter Z087 [counts/s] - priority<br>
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
    # Column 34: Chi squared of the fitted model, filter W149<br>
    # Column 34: Chi squared of the fitted model, filter Z087<br>
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
    
    model_data = {}
    header = []
    
    for i,line in enumerate(file_lines):

        items = line.replace('\n','').split()
        
        if line[0:1] == '#':
            
            header = items[2:]
            
        elif len(line.replace('\n','')) > 0:
            
            modelID = items[0]
            if len(str(modelID).split('_')) > 2:
                modelID = '_'.join(str(modelID).split('_')[0:2])
            
            entry = EventEntry({'idx': (len(model_data)+1), 
                                'modelID': modelID, 
                                'model_class': items[1]})
    
            values = []
            
            for j,f in enumerate(items[2:]):
                
                if 'None' in str(f) or len(str(f).replace('-','')) == 0:
                    
                    values.append(np.nan)
                    
                elif str(f).isalpha():
                    
                    values.append(str(f))
                    
                else:
                    
                    try:
                        
                        values.append(float(f))
                        setattr(entry,header[j],float(f))
                        
                    except ValueError:
                        
                        values.append(str(f))
                        setattr(entry,header[j],f)
            
            if page:
                print(entry.summary())
                cont = input('Continue? ')
            
            #entry.self_check()
            
            if entry.piEE != None and entry.piEN != None:
                entry.piE = np.sqrt( entry.piEE*entry.piEE + entry.piEN*entry.piEN )
                entry.sig_piE = np.sqrt( (entry.sig_piEE*entry.sig_piEE) + \
                                        (entry.sig_piEN*entry.sig_piEN) )
            
            if entry.t0 != None and entry.t0 < 2450000.0:
                entry.t0 = entry.t0 + 2450000.0
            
            # Catch for entries which included only a single passband of fs, fb
            # -> Assigned to W149 filter
            if 'fs' in dir(entry) and entry.fs_W == None and entry.fs_Z == None:
                entry.fs_W = entry.fs
                
            if 'sig_fs' in dir(entry) and entry.sig_fs_W == None and entry.sig_fs_Z == None:
                entry.sig_fs_W = entry.sig_fs
                
            if 'fb' in dir(entry) and entry.fb_W == None and entry.fb_Z == None:
                entry.fb_W = entry.fb
                
            if 'sig_fb' in dir(entry) and entry.sig_fb_W == None and entry.sig_fb_Z == None:
                entry.sig_fb_W = entry.sig_fb
                
            if entry.modelID in model_data.keys():
                model_data[entry.modelID].append(entry)
            else:
                model_data[entry.modelID] = [entry]
            
            #if len(values) == 39:
                
            #    model_data[entry.modelID] = entry
            
            #else:
                
            #    print('Error parsing line '+str(i)+' in '+path.basename(file_path))
                
    return model_data

def read_master_table(file_path,page=False):
    """Function to read the input file of the original simulation parameters
    per event"""
    
    if path.isfile(file_path) == False:
        raise IOError('Cannot find input file '+file_path)
        exit()
        
    lines = open(file_path,'r').readlines()
    
    master_data = {}
    
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
            
            true_fs_W = float(entries[57])
            true_fl_W = float(entries[63])
            
            true_fs_Z = float(entries[55])
            true_fl_Z = float(entries[61])
            
            e.t0 = float(entries[32]) + 2458234.0
            e.sig_t0 = 0.0
            e.tE = float(entries[33]) 
            e.sig_tE = 0.0
            e.u0 = float(entries[30])
            e.sig_u0 = 0.0
            e.rho = float(entries[37])
            e.sig_rho = 0.0
            e.piE = float(entries[36])     # piE -> piEN, piEE
            e.sig_piE = 0.0
            e.piEE = None
            e.sig_piEE = None
            e.piEN = None
            e.sig_piEN = None
            e.fs_W = float(entries[66])       # -> 67 for second filter
            e.sig_fs_W = 0.0
            e.fb_W = calc_fb(true_fs_W, e.fs_W, true_fl_W)
            e.sig_fb1 = None
            e.fs_Z = float(entries[67])       # -> 67 for second filter
            e.sig_fs_Z = 0.0
            e.fb_Z = calc_fb(true_fs_Z, e.fs_Z, true_fl_Z)
            e.sig_fb_Z = None
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
            
            phase0 = float(entries[45]) * (np.pi/180.0) # deg -> rads
            inc = float(entries[44]) * (np.pi/180.0) # deg -> rads
            period = float(entries[48]) * 365.24 # years -> days
            RE = float(entries[34])
            a = float(entries[43])/RE       # AU -> thetaE
            alpha0 = float(entries[31]) * (np.pi/180.0) # deg -> rads
            t_ref = float(entries[32])
            t = t_ref - 1.0
            
            (dsdt, dalphadt) = calc_orbital_parameters(phase0, inc, period, 
                                                        a, e.q, alpha0, 
                                                        t_ref, t)
            e.dsdt = dsdt
            e.dadt = dalphadt
            
            for key, icol in pars.items():
    
                if key in ['model_class']:
                    setattr(e, key, icol)
                elif key in ['idx']:
                    setattr(e, key, int(entries[icol]))
                else:
                    setattr(e, key, float(entries[icol]))
                
                if key == 'idx':
                    model_id = 'ulwdc1_'+add_lead_zeros(entries[icol],3)
                    setattr(e, 'modelID', model_id)
            
            # Remove binary parameter from PSPL events, unnecessarily 
            # present in the input data file:
            if e.model_class == 'PSPL' or e.model_class == 'CV':
                e.s = None
                e.sig_s = None
                e.q = None
                e.sig_q = None
                e.alpha = None
                e.sig_alpha = None
                e.dsdt = None
                e.sig_dsdt = None
                e.dadt = None
                e.sig_dadt = None
                e.M2 = None
                e.sig_M2 = None
                e.aperp = None
                e.sig_aperp = None
                
            master_data[model_id] = e
            
            if page:
                print(e.summary())
                cont = input('Continue? ')
            
    return master_data

def calc_fb(true_fs, blend_fs, true_fl):
    """Function to calculate the blended flux from the blended and true fluxes
    of the lens and source"""
    
    true_fb = true_fs / (blend_fs*blend_fs*(true_fs + true_fl))
    
    return true_fb
    

def calc_orbital_parameters(phase, inc, period, a, q, alpha0, t_ref, t,
                            verbose=False):
    """Method to calculate ds/dt and dalpha/dt from circular orbital 
    parameters
    All input angles are expected to be in radians
    All input times are expected to be in days
    Separation, a, is expected to be normalised by thetaE
    """
    
    phi = (np.pi/180.0)*phase + (t-t_ref)/period
    phi_ref = (np.pi/180.0)*phase
    if verbose: print('Phi = '+str(phi)+', phi_ref = '+str(phi_ref))
    
    a1 = q/(1+q)*a
    a2 = a - a1
    if verbose: print('a1 = '+str(a1)+', a2 = '+str(a2))
    
    cosinc = np.cos(inc)

    # Lens host star position
    x1 = a1 * np.cos(phi + np.pi)
    y1 = a1 * np.sin(phi + np.pi) * cosinc
    x1_ref = a1 * np.cos(phi_ref + np.pi)
    y1_ref = a1 * np.sin(phi_ref + np.pi) * cosinc
    if verbose: 
        print('Lens host star positions')
        print('x1,y1 = '+str(x1)+', '+str(y1))
        print('x1,y1 ref = '+str(x1_ref)+', '+str(y1_ref))
    
    # Companion position
    x2 = a2 * np.cos(phi)
    y2 = a2 * np.sin(phi) * cosinc
    x2_ref = a2 * np.cos(phi_ref)
    y2_ref = a2 * np.sin(phi_ref) * cosinc
    if verbose: 
        print('Lens companion positions')
        print('x2,y2 = '+str(x2)+', '+str(y2))
        print('x2,y2 ref = '+str(x2_ref)+', '+str(y2_ref))
    
    s = np.sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) )
    s_ref = np.sqrt( (x2_ref-x1_ref)*(x2_ref-x1_ref) + \
                     (y2_ref-y1_ref)*(y2_ref-y1_ref) )
    if verbose: print('s = '+str(s)+', s_ref = '+str(s_ref))
    
    dsdt = ((s - s_ref)/(t-t_ref))*365.24   # Rate of change per day -> year
    
    alpha = alpha0 - np.arctan2(y2,x2)
    alpha_ref = alpha0 - np.arctan2(y2_ref,x2_ref)
    if verbose: print('alpha = '+str(alpha)+', alpha_ref = '+str(alpha_ref))
    
    dalphadt = ((alpha - alpha_ref)/(t-t_ref))*365.24 # Per day -> year
    
    return dsdt, dalphadt
    
def add_lead_zeros(number,dp):
    """Function to zero pad a given number, to the number of characters
    indicated"""
    
    snumber = str(number)
    
    while len(snumber) < dp:
        snumber = '0'+snumber
    
    return snumber
    
if __name__ == '__main__':
    
    if len(argv) < 4:
        
        file_path = input('Please enter the path to the table file: ')
        opt = input('Is this a master table? Y or n: ')
        page = input('Page through entries? Y or n: ')
        
    else:
        
        file_path = argv[1]
        opt = argv[2]
        page = argv[3]
    
    if 'Y' in page:
        page = True
    else:
        page = False
        
    if 'Y' in opt or 'master' in opt:
        model_data = read_master_table(file_path,page=page)
    else:
        model_data = read_standard_ascii_DC_table(file_path,page=page)
    
    counts = {}
    
    for i,model_list in model_data.items():
        
        if 'Y' in opt or 'master' in opt:
            model_list = [ model_list ]
        else:
            pass
        for m in model_list:
            print(m.summary())
            
            if m.model_class in counts.keys():
                n = counts[m.model_class] + 1
                counts[m.model_class] = n
            else:
                counts[m.model_class] = 1
    
    for key, value in counts.items():
        print(str(value)+' of '+key+' class')
        