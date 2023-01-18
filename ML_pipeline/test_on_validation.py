import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, accuracy_score, precision_score, \
    recall_score


datasets_data = pd.concat([pd.read_csv("/Users/noa/Downloads/5481529/Supplementary_table_S2.txt", sep = '\t')])
datasets_data = datasets_data.drop(columns=['Published'])
datasets_data["Max_LL_score_data"] = datasets_data.max(axis=1)
#datasets_data["var_LL_score_data"] = datasets_data.var(axis=1) #RAxML-10(RT)
datasets_data["RAxML_succeeded"] = ((datasets_data["RAxML-10(RT)"]>=datasets_data["Max_LL_score_data"]-0.1))
#features_data = pd.read_csv("/Users/noa/Workspace/raxml_deep_learning_results/new_grouping_test/all_features.tsv", sep = '\t')
#features_data_validation = features_data.loc[features_data["msa_path"].str.contains("Single_gene_PROTEIN")]
validation_data_best_ll= pd.read_csv("/Users/noa/Downloads/val_best_msa_ll.tsv",sep='\t')
validation_data = validation_data_best_ll[["msa_path","best_msa_ll"]].drop_duplicates().sort_values('msa_path')
validation_data["Dataset_name"] = validation_data["msa_path"].apply(lambda x: x.split('_')[-1])
validation_data["Gene_name_tmp"] = validation_data['msa_path'].apply(lambda x: x.split('/')[-1])
validation_data["Gene_name"] = validation_data['Gene_name_tmp'].apply(lambda x: x[:x.rfind('_')])
validation_data["Gene_name_final"] = validation_data['Gene_name'].apply(lambda x: x[:x.find('.')])
validation_data_predictions = pd.read_csv('/Users/noa/Workspace/raxml_deep_learning_results/new_grouping_test/groups_run_local_test/val_no_opt.tsv', sep = '\t')
validation_data_predictions["Dataset_name"] = validation_data_predictions["msa_path"].apply(lambda x: x.split('_')[-1])
validation_data_predictions["Gene_name_tmp"] = validation_data_predictions['msa_path'].apply(lambda x: x.split('/')[-1])
validation_data_predictions["Gene_name"] = validation_data_predictions['Gene_name_tmp'].apply(lambda x: x[:x.rfind('_')])
validation_data_predictions["Gene_name_final"] = validation_data_predictions['Gene_name'].apply(lambda x: x[:x.find('.')])

mean_validation_data_predictions_per_MSA = validation_data_predictions.groupby(['Dataset_name','Gene_name_final']).agg(mean_pred = ('calibrated_prob',np.mean)).reset_index()
comp =mean_validation_data_predictions_per_MSA.merge(datasets_data, right_on = ['Data set','Gene name'], left_on = ['Dataset_name','Gene_name_final'])
print(len(comp.index))
print(roc_auc_score(comp["RAxML_succeeded"],comp["mean_pred"]))
print(average_precision_score(comp["RAxML_succeeded"],comp["mean_pred"]))
ax = sns.boxplot(data=comp, y="mean_pred",x="RAxML_succeeded",  dodge=False)
ax.set(xlabel='RAxML reachd global max', ylabel='Mean probabiltiy to reach global max')
#comp.boxplot(x="mean_pred",y="RAxML_succeeded")
plt.show()
pass




