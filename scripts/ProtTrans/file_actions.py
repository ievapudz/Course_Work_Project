import numpy
import torch

def load_NPZ_file(NPZ_file, keywords_torch, keywords_other=None):
    '''
    Loading embeddings from an NPZ file.
    NPZ_file - STRING that identifies the file to read
    keywords - LIST of keywords that identify what to extract from an NPZ
    '''

    dataset = {}
    with numpy.load(NPZ_file, allow_pickle=True) as data_loaded:
        for i in range(len(keywords_torch)):
            dataset[keywords_torch[i]] = torch.from_numpy(
                data_loaded[keywords_torch[i]])
        if(keywords_other):
            for i in range(len(keywords_other)):
                dataset[keywords_other[i]] = data_loaded[keywords_other[i]]
    return dataset
