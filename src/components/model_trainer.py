from dataclasses import dataclass
import os , sys
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor , AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from src.utils import evaluate_models , save_obj

@dataclass
class ModelTrainerConfig:
    trained_model_path : str = os.path.join('artifacts' , 'model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self , train_data , test_data):
        try:
            X_train , Y_train , X_test , Y_test = (train_data[:,:-1] , train_data[: , -1], test_data[:,:-1] , test_data[: , -1])
            models = {
                "Random Forest" : RandomForestRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Gradient Boosting' : GradientBoostingRegressor(),
                'Linear Regression' : LinearRegression(),
                'XGBRegressor' : XGBRegressor(),
                'CatBoosting Regressor' : CatBoostRegressor(verbose=0),
                'Adaboost Regressor' : AdaBoostRegressor()
            }

            params = {
                'Random Forest': {
                    'n_estimators': [10, 50, 100],
                    'criterion': ['squared_error', 'absolute_error'],
                    'max_depth': [None, 10, 20]
                },
                'Decision Tree': {
                    'criterion': ['squared_error', 'absolute_error'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20]
                },
                'Gradient Boosting': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'loss': ['squared_error', 'absolute_error']
                },
                'Linear Regression': {
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                },
                'XGBRegressor': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 6]
                },
                'CatBoosting Regressor': {
                    'iterations': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'depth': [4, 6]
                },
                'Adaboost Regressor': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'loss': ['linear', 'square']
                }
            }

            evaluate_dict = evaluate_models(models , X_train , X_test , Y_train , Y_test , params)

            best_model_r2 = max(evaluate_dict.values())
            logging.info('Finding out the best model')
            best_model = models[
                list(evaluate_dict.keys())[
                list(evaluate_dict.values()).index(best_model_r2)
                ]]
            
            if best_model_r2 < .6:
                raise CustomException('None of the model are good enough')
            
            logging.info('Saving the best model')

            save_obj(best_model , self.model_trainer_config.trained_model_path)

            return best_model_r2

        except Exception as e:
            raise CustomException(e,sys)
        
    