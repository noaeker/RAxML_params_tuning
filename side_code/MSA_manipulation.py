from side_code.config import *
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np
import logging
import random
import re
import pandas as pd
import re

def get_msa_name(msa_path, general_msa_dir):
    return msa_path.replace(general_msa_dir, "").replace(os.path.sep,
                                                         "_")


def get_msa_data(msa_path, msa_suffix):
    with open(msa_path) as original:
        msa_data = list(SeqIO.parse(original, msa_suffix))
    return msa_data


def extract_file_type(path, change_format=False, ete=False):
    filename, file_extension = os.path.splitext(path)
    if change_format:
        if file_extension == '.phy':
            file_extension = 'iphylip' if ete == True else 'phylip-relaxed'
        elif file_extension == ".fasta":
            file_extension = 'fasta'
        elif file_extension == ".nex":
            file_extension = 'nexus'
    return file_extension



def remove_gaps_and_trim_locis(sample_records, max_n_loci, loci_shift):
    all_data = np.array([list(record.seq) for record in sample_records])
    count_gaps_per_column = np.count_nonzero(((all_data == "-") | (all_data == "X")), axis=0)
    non_gapped_data = all_data[:, count_gaps_per_column < all_data.shape[0]]
    loci_trimmed_data = non_gapped_data[:, loci_shift:loci_shift + max_n_loci]
    new_sampled_records = []
    for i, old_record in enumerate(sample_records):
        sampled_record = SeqRecord(Seq("".join(list(loci_trimmed_data[i, :]))), id=old_record.id, name=old_record.name,
                                   description=old_record.description)
        new_sampled_records.append(sampled_record)
    return new_sampled_records


def trim_n_seq(original_seq_records, number_of_sequences, seed):
    seq_trimmed_seq_records = []
    seq_values = set()
    random.seed(seed)
    random.shuffle(original_seq_records)
    for record in original_seq_records:
        if len(seq_trimmed_seq_records) >= number_of_sequences:
            break
        if str(record.seq) in seq_values:
            continue
        else:
            sampled_record = SeqRecord(record.seq, id=record.id, name=record.name,
                                       description=record.description)
            seq_values.add(str(record.seq))
            seq_trimmed_seq_records.append(sampled_record)
    return seq_trimmed_seq_records


def count_unique_n_seq(original_seq_records):
    seq_values = set()
    for record in original_seq_records:
        seq = np.array(list(record.seq))
        undetermined_deq = seq[(seq == "-") | (seq == "X")]
        if len(undetermined_deq) < len(seq):
            seq_values.add("".join(seq))
    return len(seq_values)


def trim_MSA(original_alignment_path, trimmed_alignment_path, number_of_sequences,max_n_loci, loci_shift):
    original_alignment_data = get_alignment_data(original_alignment_path)
    obtained_n_seq = -1
    i = 0
    while obtained_n_seq < number_of_sequences and i <= 100:
        seq_trimmed_seq_records = trim_n_seq(original_alignment_data, number_of_sequences, seed=SEED + i)
        loci_trimmed_seq_records = remove_gaps_and_trim_locis(seq_trimmed_seq_records, max_n_loci, loci_shift)
        obtained_n_seq = count_unique_n_seq(loci_trimmed_seq_records)
        i = i + 1
    logging.debug("obtained {obtained_n_seq} sequences after {i} iterations!".format(obtained_n_seq=obtained_n_seq, i=i))
    try:
        SeqIO.write(loci_trimmed_seq_records, trimmed_alignment_path, 'fasta')
        logging.debug(" {} sequences written succesfully to new file {}".format(len(seq_trimmed_seq_records),
                                                                               trimmed_alignment_path))
    except:
        logging.error("ERROR! {} sequences NOT written succesfully to new file {}".format(number_of_sequences,
                                                                                          trimmed_alignment_path))



def remove_MSAs_with_not_enough_seq(file_path_list, min_seq):
    proper_file_path_list = []
    for path in file_path_list:
        file_type_biopython = extract_file_type(path, True)
        with open(path) as file:
            n_seq = len(list(SeqIO.parse(file, file_type_biopython)))
            if n_seq >= min_seq:
                proper_file_path_list.append(path)
    return proper_file_path_list



def get_positions_stats(alignment_df):
    alignment_df_fixed = alignment_df.replace('-', np.nan)
    gap_positions_pct = np.mean(alignment_df_fixed.isnull().sum() / len(alignment_df_fixed))
    counts_per_position = [dict(alignment_df_fixed[col].value_counts(dropna=True)) for col in list(alignment_df)]
    probabilities = [list(map(lambda x: x / sum(counts_per_position[col].values()), counts_per_position[col].values()))
                     for col in
                     list(alignment_df)]
    entropy = [sum(list(map(lambda x: -x * np.log(x), probabilities[col]))) for col in list(alignment_df)]
    avg_entropy = np.mean(entropy)
    constant_sites_pct = sum([1 for et in entropy if et == 0]) / len(entropy)
    return constant_sites_pct, avg_entropy, gap_positions_pct


def get_alignment_data(msa_path):
    with open(msa_path) as file:
        try:
            file_type_biopython = extract_file_type(msa_path, True)
            data = list(SeqIO.parse(file, file_type_biopython))
        except:
            try:
                data = list(SeqIO.parse(file, 'fasta'))
            except:
                try:
                    data = list(SeqIO.parse(file, 'phylip-relaxed'))
                except:
                    return -1
        if len(data) == 0:
            return -1
        return data


def alignment_list_to_df(alignment_data):
    alignment_list = [list(alignment_data[i].seq) for i in range(len(alignment_data))]
    loci_num = len(alignment_data[0].seq)
    columns = list(range(0, loci_num))
    original_alignment_df = pd.DataFrame(alignment_list, columns=columns)
    return original_alignment_df

def remove_env_path_prefix(path):
    path = path.replace("/groups/pupko/noaeker/", "")
    path = path.replace("/Users/noa/Workspace/","")
    return  path


def remove_MSAs_with_not_enough_seq_and_locis(file_path_list, min_n_seq, max_n_seq, min_n_loci):
    proper_file_path_list = []
    for path in file_path_list:
        data = get_alignment_data(path)
        if data==-1:
            continue
        n_seq = len(data)
        n_loci = len(data[0])
        if n_seq >= min_n_seq and n_seq<=max_n_seq and n_loci>= min_n_loci:
            proper_file_path_list.append(path)
    return proper_file_path_list



def get_msa_type(msa_path):
    #msa_path_no_extension = os.path.splitext(msa_path)[0]
    #if re.search('\w+D[\da-z]+', msa_path_no_extension.split(os.sep)[-2]) is not None:
    #    msa_type = "DNA"
    #else:
    #    msa_type = "AA"
    #return msa_type
    all_letters = ""
    for record in  get_alignment_data(msa_path):
        all_letters = all_letters+str(record.seq)
    letters_str = all_letters.replace("-","")
    ACGT_content = re.sub('[ACGTNOX?]','',letters_str)
    if len(ACGT_content)/len(letters_str)>0.9:
        return "DNA"
    else:
        return "AA"

