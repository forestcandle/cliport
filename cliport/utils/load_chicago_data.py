import pandas as pd
import os
from pyglet.resource import file


def load_chicago_dataset(chicago_datasets_root=None):
    ''' Load chicago face dataset.

    args:
        chicago_datasets_root: str, path to cfd folder

    returns:
        face_dict: dict, key is each model ID, each entry is
        a dict with full file path to face image, and self-id
        ethnicity and gender.

    '''
    if chicago_datasets_root is None:
        chicago_datasets_root = os.getcwd()
    if 'CFD Version 3.0' not in chicago_datasets_root:
        chicago_data_v3 = os.path.join(chicago_datasets_root, 'CFD Version 3.0')
        if os.path.exists(chicago_data_v3):
            chicago_datasets_root = chicago_data_v3
    workbook_path = os.path.join(chicago_datasets_root, "CFD 3.0 Norming Data and Codebook.xlsx")
    if not os.path.exists(workbook_path):
        raise FileNotFoundError('Could not find excel workbook: ' + str(workbook_path))
    workbook = pd.read_excel(workbook_path, sheet_name=1).values
    face_dict={}
    identity_dict={}
    for row_ind in range(8, workbook.shape[0]):
        face_file_folder=os.path.join(chicago_datasets_root, "Images/CFD", workbook[row_ind][0])
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
                          "gender": gender,
                          "face_name": face_file_name}
        if ethnicity not in identity_dict:
            identity_dict[ethnicity]={}
        if gender not in identity_dict[ethnicity]:
            identity_dict[ethnicity][gender]=[]
        identity_dict[ethnicity][gender].append({"face_file" : full_face_file_name, "face_name": face_file_name})

    return face_dict, identity_dict


if __name__ == '__main__':
    face_dict = load_chicago_dataset('/Users/athundt/Downloads/chicago face database/CFD Version 3.0')
    print(face_dict)