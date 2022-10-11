from side_code.config import *
from side_code.file_handling import create_dir_if_not_exists
import re
import os



def main():
    pandit_database_path = '/Users/noa/Workspace/data/Pandit'
    formatted_msas_folder = '/Users/noa/Workspace/data/New_MSAs/Pandit_msas'
    create_dir_if_not_exists(formatted_msas_folder)
    with open(pandit_database_path) as PANDIT_DATABASE:
        data = PANDIT_DATABASE.read()
    raw_data_per_MSA = re.split(pattern=r'FAM\s+\S+', string=data)[1:]
    print(f"Number of MSAs: {len(raw_data_per_MSA)}")
    MSA_names = re.findall(pattern=r'FAM\s+(\S+)', string=data)
    data_per_MSA = {}
    for MSA, MSA_name in zip(raw_data_per_MSA,MSA_names):
        MSA_species = re.findall('NAM\s+(\S+)\s*ASQ\s*(\S+)',MSA)
        MSA_path = os.path.join(formatted_msas_folder,MSA_name+".fasta")
        MSA_content = "\n".join([">"+name+"\n"+sequence for name,sequence in MSA_species])
        MSA_data = {"n_species": len(MSA_species), "n_loci": len(MSA_species[0][1])}
        data_per_MSA[MSA_name] = MSA_data
        with open(MSA_path,'w') as MSA_FILE:
            MSA_FILE.write(MSA_content)
    print(len([x for x in data_per_MSA if data_per_MSA[x]['n_species']>30 and data_per_MSA[x]['n_loci']>100]))


if __name__ == "__main__":
    main()

