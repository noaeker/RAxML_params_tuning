from help_functions import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', action='store', type=str)
    parser.add_argument('--n_jobs', action='store', type=int,)
    parser.add_argument('--output_folder', action='store', type=str)
    args = parser.parse_args()
    csvs_path_list = [f'{args.folder}/{n}.{CSV_SUFFIX}' for n in range(args.n_jobs)]
    output_csv_path = os.path.join(args.output_folder, "all_jobs_tmp_results.tsv")
    add_csvs_content(csvs_path_list,output_csv_path)



if __name__ == "__main__":
    main()