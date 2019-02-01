import json
import os

def collect_data(cell_range, filename):
    save_data(collate_output(cell_range, filename), filename)


def save_data(data, filename):
    """Dumps data to json.

    """
    with open((os.getcwd() + "/results/{0}.txt").format(filename), 'w') as f:
        json.dump(data, f)

def save_cell_data(data, filename, cell):
    """Dumps cell data to json.

    """
    with open((os.getcwd() + "/results/{0}_{1}.txt").format(filename, cell), 'w') as f:
        json.dump(data, f)

def collate_output(cell_range, filename):
    out = {cell: get_data(filename, cell) for cell in range(*cell_range)}
    return out
 

def get_data(filename, cell):
    with open((os.getcwd() + "/results/{0}_{1}.txt").format(filename, cell)) as d:
        data = json.load(d)
    return data[str(cell)]