import os
import shutil



dna_folder = '/Users/noa/Workspace/data/single-gene_alignments/DNA'
destination_folder = '/Users/noa/Workspace/data/New_MSAs/Single_gene_DNA/data'
if os.path.exists(dna_folder):
    for sub_dir in os.listdir(dna_folder):
        sub_dir_path = os.path.join(dna_folder, sub_dir)
        if os.path.isdir(sub_dir_path):
            for file in os.listdir(sub_dir_path)[:50]:
                file_name = file + "_" + str(sub_dir)
                file_path = os.path.join(sub_dir_path, file)
                print(file_path)
                shutil.copyfile(file_path, os.path.join(destination_folder, file_name))



protein_folder = '/Users/noa/Workspace/data/single-gene_alignments/PROTEIN'
destination_folder = '/Users/noa/Workspace/data/New_MSAs/Single_gene_PROTEIN/data'
if os.path.exists(protein_folder):
    for sub_dir in os.listdir(protein_folder):
        sub_dir_path = os.path.join(protein_folder, sub_dir)
        if os.path.isdir(sub_dir_path):
            for file in os.listdir(sub_dir_path)[:50]:
                file_name = file + "_" + str(sub_dir)
                file_path = os.path.join(sub_dir_path, file)
                print(file_path)
                shutil.copyfile(file_path, os.path.join(destination_folder, file_name))

