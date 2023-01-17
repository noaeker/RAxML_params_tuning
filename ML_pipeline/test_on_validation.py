import pandas as pd
import re



datasets_data = pd.read_csv("/Users/noa/Downloads/5481529/Supplementary_table_S15.txt", sep = '\t')
datasets_data = datasets_data.drop(columns=['Published'])
datasets_data["max_LL_score_data"] = datasets_data.max(axis=1)
Chen = datasets_data.loc[datasets_data["Data set"]=='ChenA4'][["Gene name","max_LL_score_data"]].sort_values('Gene name')
#features_data = pd.read_csv("/Users/noa/Workspace/raxml_deep_learning_results/new_grouping_test/all_features.tsv", sep = '\t')
#features_data_validation = features_data.loc[features_data["msa_path"].str.contains("Single_gene_PROTEIN")]
validation_data= pd.read_csv("/Users/noa/Downloads/val_best_msa_ll.tsv",sep='\t')
Chen_validation_data = validation_data[["msa_path","best_msa_ll"]].loc[validation_data['msa_path'].str.contains('Chen')].drop_duplicates().sort_values('msa_path')
Chen_validation_data["Gene name"] = Chen_validation_data['msa_path'].apply(lambda x: re.search('Pro_\w+',x).group() )
res = Chen_validation_data.merge(Chen, on = "Gene name")
pass
