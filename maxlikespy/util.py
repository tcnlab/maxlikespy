import json
import os


def collect_data(cell_range, filename, path):
    save_data(collate_output(cell_range, filename, path), filename, path)

def save_data(data, filename, path=None, cell=None):
    """Dumps data to json.

    """
    save_path = check_path(path)
    print("saving {0} to {1}".format(filename, save_path))

    if cell is not None:
        with open((save_path + "/results/{0}_{1}.json").format(filename, cell), 'w') as f:
            json.dump(data, f, sort_keys=False, indent=4, separators=(',', ': '))
    else:
        with open((save_path + "/results/{0}.json").format(filename), 'w') as f:
            json.dump(data, f, sort_keys=False, indent=4, separators=(',', ': '))

def collate_output(cell_range, filename, path):
    out = {cell: get_data(filename, cell, path) for cell in cell_range}
    # out["log"] = get_data(filename, cell_range[0])["log"]
    return out
 
def get_data(filename, cell, path): #add try 
    with open((path + "/results/{0}_{1}.json").format(filename, cell)) as d:
        data = json.load(d)
    return data[str(cell)]

def update_comparisons(cell, model, result, run_log, path=None, odd_even=False):
    save_path = check_path(path)  
    print("saving model comparisons to {0}".format(save_path))
    if odd_even and type(odd_even) == str:
        path = save_path + "/results/model_comparisons_{0}_{1}.json".format(odd_even, cell)
    else:
        path = save_path + "/results/model_comparisons_{0}.json".format(cell)
    if os.path.exists(path):
        with open(path, 'r') as f:
            d = json.load(f)
            d[cell][model] = result
    else:
        d = {"log":run_log, cell:{model:result}}
    with open(path, 'w') as f:
        json.dump(d, f, sort_keys=False, indent=4, separators=(',', ': '))

def check_path(path):
    if path:
        save_path = path
    else:
        save_path = os.getcwd()
    os.makedirs(save_path+"/results/figs/", mode=0o777, exist_ok=True)   

    return save_path

# def embed_log():

    

