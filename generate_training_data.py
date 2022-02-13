from help_functions import *


def extract_SPR_features(msa_path):
    pass

def extract_general_msa_features(msa_path):
    alignment_data = get_alignment_data(msa_path)
    alignment_df = alignment_list_to_df(alignment_data)

    return alignment_df



def main():
        msa_path = "/Users/noa/Workspace/data/single-gene_alignments/XiD4/Cluster3035.nt.aln"
        print(extract_general_msa_features(msa_path))

if __name__ == "__main__":
    main()






