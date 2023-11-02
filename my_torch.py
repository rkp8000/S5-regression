import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

cc = np.concatenate


def song_to_x(song):
    i_s = (np.array([note=='S' for note in song]))
    i_p = (np.array([note=='P' for note in song]))
    return cc([i_s, i_p])
    
def fit_song_fmtn(paths, alpha):
    ys = {}
    xs = {}
    
    splits = ['train', 'eval', 'test']
    
    for split in splits:
        df = pd.read_csv(paths[split], sep='\t', header=None)

        ys[split] = np.array(df[0])
        xs[split] = np.array([song_to_x(song) for song in df[3]])
        
    rgr = Ridge(alpha=alpha).fit(xs['train'], ys['train'])
    
    y_hats = {split: rgr.predict(xs[split]) for split in splits}
    
    return ys, y_hats
