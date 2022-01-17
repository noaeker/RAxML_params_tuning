import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_path = "/Users/noa/Workspace/raxml_deep_learning_results/0.tsv"
    data = pd.read_csv(csv_path, sep = '\t')
    print(data.columns)
    plt.scatter(data["spr_cutoff"], data['delta_ll_from_overall_msa_best_topology'])
    plt.scatter(data["spr_cutoff"], y)



if __name__ == "__main__":
    main()