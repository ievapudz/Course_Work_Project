import os
from Bio import SeqIO
from Bio.Seq import Seq

def filter_FASTA(input_FASTA, length_threshold):
    """
    Filtering FASTA records by the length (before generation of embeddings)
    input_FASTA - STRING a path to the input FASTA file
    length_threshold  - INT max length of the protein sequence
    returns STRING of the location of filtered FASTA file
    """
    filtered_records = []
    for record in SeqIO.parse(input_FASTA, "fasta"):
        if(len(record.seq) <= length_threshold):
            filtered_records.append(record)

    filtered_FASTA = './'+os.path.basename(os.path.splitext(input_FASTA)[0])+\
                     '_filtered_by_'+str(length_threshold)+'.fasta'
    file_handle = open(filtered_FASTA, 'w')
    for record in filtered_records:
        modified_id = str(record.id).replace('-', '_')
        file_handle.write('>'+modified_id+'\n'+str(record.seq)+'\n')
    file_handle.close()
    return filtered_FASTA
