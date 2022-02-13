import sys
import pandas as pd
import argparse
import os

def unify_csvs(csvs_path_list, unified_csv_path):
    tmp_dataframes = []
    for csv_path in range(len(csvs_path_list)):
        tmp_df = pd.read_csv(csv_path,sep='\t')
        print("size = "+ str(len(tmp_df.index)))
        tmp_dataframes.append(tmp_df)
    combined_df = pd.concat(tmp_dataframes, sort=False)
    combined_df.to_csv(unified_csv_path,sep = '\t')
    return combined_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', action='store', type=str)
    parser.add_argument('--n_jobs', action='store', type=int)
    parser.add_argument('--output_folder', action='store', type=str)
    args = parser.parse_args()
    csvs_path_list = [f'{args.folder}/job_{n}/{n}.tsv' for n in range(args.n_jobs)]
    output_csv_path = os.path.join(args.output_folder, "all_jobs_tmp_results.tsv")
    unify_csvs(csvs_path_list,output_csv_path)



if __name__ == "__main__":
    main()