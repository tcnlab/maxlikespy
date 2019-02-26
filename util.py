import json
import os

def collect_data(cell_range, filename):
    save_data(collate_output(cell_range, filename), filename)

def save_data(data, filename, path="", cell=None):
    """Dumps data to json.

        """
    if cell:
        with open((os.getcwd() + "/results/{0}_{1}.txt").format(filename, cell), 'w') as f:
            json.dump(data, f)
    else:
        with open((os.getcwd() + path + "/{0}.json").format(filename), 'w') as f:
                json.dump(data, f)
                
def collate_output(cell_range, filename):
    out = {cell: get_data(filename, cell) for cell in cell_range}
    return out
 
def get_data(filename, cell): #add try 
    with open((os.getcwd() + "/results/{0}_{1}.txt").format(filename, cell)) as d:
        data = json.load(d)
    return data[str(cell)]