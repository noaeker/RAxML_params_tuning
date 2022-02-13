import os
from help_functions import *

def unify_csvs(csvs_path_list, unified_csv_path):
    tmp_dataframes = []
    for csv_path in csvs_path_list:
        tmp_df = pd.read_csv(csv_path,sep=CSV_SEP)
        print("size = "+ str(len(tmp_df.index)))
        tmp_dataframes.append(tmp_df)
    combined_df = pd.concat(tmp_dataframes, sort=False)
    combined_df.to_csv(unified_csv_path,sep = CSV_SEP)
    return combined_df

def main():
    general_folder = "/Users/noa/Workspace/raxml_deep_learning_results/single_gene_MSAs_new"
    overall_data_path = "/Users/noa/Workspace/raxml_deep_learning_results/single_gene_MSAs_new/full_raxml_data.tsv"
    csv_paths= [f"{general_folder}/job_{i}/{i}.tsv" for i in range(10)]
    print(csv_paths)
    unify_csvs(csv_paths, overall_data_path)

if __name__ == "__main__":
    main()