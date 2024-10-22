import time

import numpy as np

class stopwatch: 
    def __init__(self, length): 
        self.length = length 
        
    def start(self): 
        self.t0 = time.time()
        
    def elapsed(self): 
        return time.time() - self.t0
    
    def time_remaining(self, i): 
        if i > 0: 
            t_elapse = self.elapsed()
            frac_done = i / self.length
            time_rem = (t_elapse / frac_done) * (1 - frac_done)
        else: 
            time_rem = np.inf
        return time_rem

def most_conf_frac(scores, frac): 
    """get topk NN predictions on the most confident fraction
    of attacked examples. Run after compute_topk_preds
    Input:
        most_conf_frac: scalar [0,1], most confident frac of examps
    Return: 
        frac_idxs: indices of the most confident examples
        preds: topk predictions of these examples 
    """
    n_most_conf = int(frac * len(scores))
        
    #get most confident subset of indices
    most_conf_idxs = np.argsort(scores)[::-1][:n_most_conf]

    #get predictions 
    most_conf_preds = scores[most_conf_idxs]

    return most_conf_idxs, most_conf_preds