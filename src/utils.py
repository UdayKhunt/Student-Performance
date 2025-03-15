from src.exception import CustomException
from src.logger import logging
import sys,os
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_obj(object , filepath):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path , exist_ok=True)

        with open(filepath , 'wb') as file_obj:
            pickle.dump(object , file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(models , X_train , X_test , Y_train , Y_test , params):
    try:
        logging.info('Evaluating models')
        evaluate_dict = {}
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            model_param = params[model_name]

            grid = GridSearchCV(model , param_grid=model_param , cv = 3)
            grid.fit(X_train , Y_train)

            model.set_params(**grid.best_params_)
            model.fit(X_train , Y_train)
            Y_pred = model.predict(X_test)
            r2 = r2_score(Y_test , Y_pred)

            evaluate_dict[model_name] = r2
        logging.info('Evaluation finished')
        return evaluate_dict
    except Exception as e:
        raise CustomException(e,sys)
    
def load_obj(filepath):
    try:
        with open(filepath , 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)