# A module that works with ProtTrans models. Functions were
# adapted from ProtTrans authors' Google Colab notebook

from transformers import T5EncoderModel, T5Tokenizer
import torch
from torch import nn
import time
import os

def get_pretrained_model(model_path):
    '''
    Fetches the model accordingly to the model_path
    model_path - STRING that identifies the model to fetch
    Returns model.
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if(os.path.exists(model_path+'/pytorch_model.bin') and
        os.path.exists(model_path+'/config.json')):
        model = T5EncoderModel.from_pretrained(model_path+'/pytorch_model.bin',
            config=model_path+'/config.json')
    else:
        model = T5EncoderModel.from_pretrained(model_path)
    model = model.to(device)
    model = model.eval()

    return model

def save_pretrained_model(model, path_to_dir):
    '''
    Saves the model to the file (PT).
    model - OBJ [PreTrainedModel] of the model to save
    path_to_dir - STRING that identifies the destination to save the model
    '''

    model.save_pretrained(path_to_dir)

def get_tokenizer(model_path):
    '''
    Fetches the tokenizer accordingly to the model_path
    model_path - STRING that identifies the model whose tokenizer 
                 should be fetched
    Returns tokenizer.
    '''
    if(os.path.exists(model_path+'/pytorch_model.bin') and
        os.path.exists(model_path+'/config.json')):
        tokenizer = T5Tokenizer.from_pretrained(model_path+'/pytorch_model.bin',
            config=model_path+'/config.json', do_lower_case=False)
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)

    return tokenizer

def process_FASTA(fasta_path, split_char="!", id_field=0):
    '''
    Reads in fasta file containing multiple sequences.
    Split_char and id_field allow to control identifier extraction from header.
    E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
    Returns dictionary holding multiple sequences or only single 
    sequence, depending on input file.
    '''

    seqs = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                seqs[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, 
                # drop gaps and cast to upper-case
                seq= ''.join( line.split() ).upper().replace("-","")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U','X').replace('Z','X').replace('O','X')
                seqs[ uniprot_id ] += seq
    example_id=next(iter(seqs))
    print("Read {} sequences.".format(len(seqs)))
    print("Example:\n{}\n{}".format(example_id,seqs[example_id]))

    return seqs

def get_embeddings(model, tokenizer, seqs, per_residue, per_protein, per_protein_std=False,
    per_protein_q=False, per_protein_hist=False, max_residues=4000, max_seq_len=1000, max_batch=100):

    '''
    Generation of embeddings via batch-processing.
    per_residue indicates that embeddings for each residue in a protein 
                should be returned.
    per_protein indicates that embeddings for a whole protein should be 
                returned (average-pooling).
    per_protein_std indicates that standard deviations of per residue embeddings
                will be returned
    per_protein_q indicated that quantiles of per residue embeddings will be
                returned
    max_residues gives the upper limit of residues within one batch.
    max_seq_len gives the upper sequences length for applying 
                batch-processing.
    max_batch gives the upper number of sequences per batch
    
    Returns results depending on the option in the input.
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    results = { 'per_res_representations' : dict(),
                'mean_representations' : dict(),
                'std_representations' : dict(),
                'quantile_representations' : dict(),
                'hist_representations' : dict() }

    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    start = time.time()
    batch = list()

    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        """
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding='longest')
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
        """
        pdb_ids, seqs, seq_lens = zip(*batch)
        batch = list()

        token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding='longest')
        input_ids = torch.tensor(token_encoding['input_ids']).to(device)
        attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

        try:
            with torch.no_grad():
                # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                embedding_repr = model(input_ids, attention_mask=attention_mask)
        except RuntimeError:
            print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
            continue

        for batch_idx, identifier in enumerate(pdb_ids):
            s_len = seq_lens[batch_idx]
            # slice off padding --> batch-size x seq_len x embedding_dim  
            emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
            if per_residue:
                # store per-residue embeddings (Lx1024)
                results["per_res_representations"][ identifier ] = emb.detach().cpu().numpy().squeeze()
            if per_protein:
                # apply average-pooling to derive per-protein embeddings (1024-d)
                protein_emb = emb.mean(dim=0)
                results["mean_representations"][identifier] = protein_emb.detach().cpu().numpy().squeeze()
            if per_protein_std: # variation from the original code
                protein_emb = emb.std(dim=0)
                results["std_representations"][identifier] = protein_emb.detach().cpu().numpy().squeeze()
            if per_protein_q: # variation from the original code
                protein_emb = emb.quantile(q=torch.Tensor([0, 0.25, 0.5, 0.75, 1]).to(device), dim=0)
                results["quantile_representations"][identifier] = protein_emb.detach().cpu().numpy().squeeze()
            if per_protein_hist: # variation from the original code
                protein_emb = emb.histc(bins=5)
                results["hist_representations"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

    passed_time=time.time()-start

    avg_time = passed_time/len(results['per_res_representations']) if per_residue else passed_time/len(results['mean_representations'])

    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["per_res_representations"])))
    print('Total number of per-protein mean embeddings: {}'.format(len(results["mean_representations"])))
    print('Total number of per-protein std embeddings: {}'.format(len(results["std_representations"]))) # variation from the original code
    print('Total number of per-protein quantile embeddings: {}'.format(len(results["quantile_representations"]))) # variation from the original code
    print('Total number of per-protein histogram embeddings: {}'.format(len(results["hist_representations"]))) # variation from the original code
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')

    return results

class FNN(nn.Module):
    '''
    Class that defines the model for subcellular localisation and 
    classification to membrane / soluble proteins classification
    '''
    def __init__(self):
        super(FNN, self).__init__()
        # Linear layer, taking embedding dimension 1024 to make predictions:
        self.layer = nn.Sequential(
            nn.Linear(1024, 32),
            nn.Dropout(0.25),
            nn.ReLU(),
        )

        # subcell. classification head
        self.loc_classifier = nn.Linear(32, 10)
        # membrane classification head
        self.mem_classifier = nn.Linear(32, 2)

    def forward(self, x):
        # Inference
        out = self.layer(x)
        Yhat_loc = self.loc_classifier(out)
        Yhat_mem = self.mem_classifier(out)
        return Yhat_loc, Yhat_mem

