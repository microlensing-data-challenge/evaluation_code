from os import path
from astropy.table import Table

def load_lc(file_path):
    """
    Function to load a lightcurve in the data challenge standard format.

    Parameters:
        file_path str  Path to lightcurve text file

    Returns:
        lc  Table   Lightcurve data table
    """

    if not path.isfile(file_path):
        raise IOError('Cannot find lightcurve file ' + file_path)

    data_table = Table.read(file_path, format='ascii')

    # Files do not contain column headers so we rename them for clarity
    data_table.rename_column('col1', 'JD')
    data_table.rename_column('col2', 'mag')
    data_table.rename_column('col3', 'mag_err')

    return data_table