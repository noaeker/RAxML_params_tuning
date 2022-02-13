from config import *
import pandas as pd
import numpy as np
from raxml import *

def main():
    overall_data_path = f"{RESULTS_FOLDER}/single_gene_MSAs/full_raxml_data.tsv"
    curr_run_directory = os.path.join(RESULTS_FOLDER,)
    data = pd.read_csv(overall_data_path, sep=CSV_SEP)
    msa_paths = list(np.unique(data["original_alignment_path"]))
    if LOCAL_RUN:
        msa_paths = [msa_path.replace("/groups/pupko/noaeker/","/Users/noa/Workspace") for msa_path in msa_paths]
    print(len(msa_paths))
    for msa_path in msa_paths:
        msa_path_no_extension = os.path.splitext(msa_path)[0]
        if re.search('\w+D[\da-z]+', msa_path_no_extension.split(os.sep)[-2]) is not None:
            msa_type = "DNA"
        else:
            msa_type = "AA"
        msa_raxml_features  =extract_raxml_statistics_from_msa(msa_path, msa_type, output_name, curr_run_directory)

if __name__ == "__main__":
    main()