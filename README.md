# RAxML_params_tuning

The entire pipeline can be divided to two parts :

A. **raw data generation- generation of 1,200 single tree runs for each MSA**\n
folder: raw_data_generation 
The main file is tp.py, in which :
1. 20 Parsimony and 20 random starting trees are generated for each MSA
2. For each starting tree, required tree-searches are formed base on a grid of tree-search parameters (i.e., SPR radius and SPR cutoff)
Finally, the requird RAxML runs are parallelized, are perfomred using the tunne_params.py file.

B. **Machine-learning pipeline on tree-searches from multiple starting points**.
folder: groups_paper_ML_code
The main file is ML_pipeline_groups_runner.py, in which:
1. The sampling of final trees, to create muliple starting point tree searches, is performed in groups_data_generation.py. First, for each MSA, the single-tree search runs from the previous section are being processed, and the estimated global maximum tree is recorded. For each MSA, at each iteration, a specific tree-search hueristic is randomly selected,
as well as the number of parsimnoy trees and random trees. Both features of the final trees and features of the MSA are extracted, and a binary outcome of reaching the global is determined.
2. The machine-learning pipeline is performed in groups_ML_pipeline.py. In this part, gradient boosting model is applied on the data generated in the previous secrion.
3. Model metrics such as AUC, precision and accuracy are estimated both on the test data and on external validaiton sets.


