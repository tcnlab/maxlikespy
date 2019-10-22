import json
import os


def collect_data(cell_range, filename):
    save_data(collate_output(cell_range, filename), filename)

def save_data(data, filename, path=None, cell=None):
    """Dumps data to json.

    """
    save_path = check_path(path)

    if cell is not None:
        with open((save_path + "/results/{0}_{1}.json").format(filename, cell), 'w') as f:
            json.dump(data, f, sort_keys=True, indent=4, separators=(',', ': '))
    else:
        with open((save_path + "/results/{0}.json").format(filename), 'w') as f:
            json.dump(data, f, sort_keys=True, indent=4, separators=(',', ': '))

def collate_output(cell_range, filename):
    out = {cell: get_data(filename, cell) for cell in cell_range}
    return out
 
def get_data(filename, cell): #add try 
    with open((os.getcwd() + "/results/{0}_{1}.json").format(filename, cell)) as d:
        data = json.load(d)
    return data[str(cell)]

def update_comparisons(cell, model, result, path=None, odd_even=False):
    save_path = check_path(path)  

    if odd_even and type(odd_even) == str:
        path = save_path + "/results/model_comparisons_{0}_{1}.json".format(odd_even, cell)
    else:
        path = save_path + "/results/model_comparisons_{0}.json".format(cell)
    if os.path.exists(path):
        with open(path, 'r') as f:
            d = json.load(f)
            d[cell][model] = result
    else:
        d = {cell:{model:result}}
    with open(path, 'w') as f:
        json.dump(d, f, sort_keys=True, indent=4, separators=(',', ': '))

def check_path(path):
    if path:
        save_path = path
    else:
        save_path = os.getcwd()

    if not os.path.exists(save_path+"/results"):
        os.mkdir(save_path+"/results")
    if not os.path.exists(save_path+"/results/figs/"):
        os.mkdir(save_path+"/results/figs/")      
    return save_path



    

