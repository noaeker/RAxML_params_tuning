from side_code.config import *
from side_code.file_handling import create_dir_if_not_exists
import re
import os
import shutil



def main():
   main_folder = '/Users/noa/Workspace/data/Dana/groups_4/itay_mayrose/danaazouri/PhyAI/submission_data/training_data'
   destination_folder = '/Users/noa/Workspace/data/New_MSAs/Treebase'
   if os.path.exists(main_folder):
       for sub_dir in os.listdir(main_folder):
           sub_dir_path = os.path.join(main_folder, sub_dir)
           if os.path.isdir(sub_dir_path) and sub_dir.isnumeric():
               for file in os.listdir(sub_dir_path):
                   file_name = file+"_"+str(sub_dir)
                   file_path = os.path.join(sub_dir_path, file)
                   print(file_path)
                   shutil.copyfile(file_path, os.path.join(destination_folder,file_name))



if __name__ == "__main__":
    main()

