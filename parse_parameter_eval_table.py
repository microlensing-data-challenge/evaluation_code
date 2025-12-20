import argparse
from parse_table1 import EventEntry
from bs4 import BeautifulSoup

def load_parameter_evaluation_table(file_path):
    """
    Function to load and parse the HTML-format parameter evaluation table for a given
    team's results, and return the contents as a dictionary of EventEntry objects.

    Parameters:
        file_path string  Full path to input HTML file

    Returned:
        model_data dict   Modeling results
    """

    model_data = {}

    # Load the HTML file
    with open(file_path) as f:
        soup = BeautifulSoup(f, features="html.parser")

    # Find the tags for the table header and create a list of the header items
    # Parse two column headers that where the headers are not the same as the
    # EventEntry parameters
    hdr = [x.contents[0] for x in soup.find_all('th')]
    sub_keys = {
        'ModelID': 'modelID',
        'Class': 'model_class',
        'ds/dt': 'dsdt',
        'dalpha/dt': 'dadt'
    }
    for key, new_key in sub_keys.items():
        idx = hdr.index(key)
        hdr[idx] = new_key

    # Now parse the table rows, creating a dictionary of the table entries in each case,
    # and using it to create a EventEntry object.
    for row in soup.find_all('tr'):
        cells = row.find_all('td')
        entries = [x.contents[0] for x in cells]
        data = {key: None for key in hdr}

        for i,heading in enumerate(hdr):
            # Some entries have both the value and uncertainty in the same string,
            # so split these.  Note some cells also have a second line, separated by a <br>
            # with the true parameter value in italics.  Inspection indicates that BeautifulSoup
            # behaves a little strangely with these, looping over the same line several times
            # but only ever taking the first row of the cell.  So the following code works.
            values = entries[i].split()
            if len(values) > 1:
                data[heading] = parse_value(values[0])
                data['sig_' + heading] = parse_value(values[-1])
            else:
                data[heading] = parse_value(entries[i])

        # Create EventEntry object
        model_data[data['modelID']] = EventEntry(data)

    return model_data

def parse_value(value):
    """
    Function to parse a value as a float if possible, or return a string or None entry
    unchanged
    """

    if 'none' in str(value).lower():
        result = value
    else:
        try:
            result = float(value)
        except ValueError:
            result = value

    return result

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Path to input HTML file')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    model_data = load_parameter_evaluation_table(args.file_path)
    for key, entry in model_data.items():
        print(entry.summary())