from src.exception import CustomException
from src.logger import logging
import sys,os
import pickle

def save_obj(object , filepath):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path , exist_ok=True)

        with open(filepath , 'wb') as file_obj:
            pickle.dump(object , file_obj)

    except Exception as e:
        raise CustomException(e,sys)