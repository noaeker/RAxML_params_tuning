import pandas as pd


def main():
    csv_path = "/Users/noa/Workspace/raxml_deep_learning_results/first_tuning/tune_raxml.csv"
    data = pd.read_csv(csv_path)
    best_results = data[data["is_best_run"]==True].copy().set_index(['msa_name','starting_tree_ind','tree_type'])
    default_results = data[data["run_name"] == "default"].copy().set_index(['msa_name','starting_tree_ind','tree_type'])
    best_vs_default = best_results.join(default_results)
    print(best_vs_default)
    print(default_results.shape)



if __name__ == "__main__":
    main()