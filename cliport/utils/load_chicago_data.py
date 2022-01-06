import pandas as pd
import os
from pyglet.resource import file


def load_chicago_dataset(chicago_datasets_root="/media/ssd_2000/cfd/"):
    ''' Load chicago face dataset.
    
    args:
        chicago_datasets_root: str, path to cfd folder
    
    returns:
        face_dict: dict, key is each model ID, each entry is
        a dict with full file path to face image, and self-id
        ethnicity and gender.
    
    '''
    
    workbook = pd.read_excel(os.path.join(chicago_datasets_root, "CFD Version 3.0/CFD 3.0 Norming Data and Codebook.xlsx"), sheet_name=1).values
    face_dict={}
    for row_ind in range(8, workbook.shape[0]):
        face_file_folder=os.path.join(chicago_datasets_root, "CFD Version 3.0/Images/CFD", workbook[row_ind][0])
        dir_contents=os.listdir(face_file_folder)
        for file in dir_contents:
            if file.endswith(".jpg"):
                face_file_name=file
                break
        full_face_file_name=os.path.join(face_file_folder, face_file_name)
        ethnicity=workbook[row_ind][1]
        gender=workbook[row_ind][2]
        face_dict[workbook[row_ind][0]]={"face_file" : full_face_file_name, 
                          "ethnicity": ethnicity, 
                          "gender": gender}
    return face_dict