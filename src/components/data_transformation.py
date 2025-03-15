from dataclasses import dataclass
import os,sys
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path : str = os.path.join('artifacts' , 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
        
            cat_pipeline = Pipeline(
                steps = [
                    ('Imputer' , SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncode' , OneHotEncoder()),
                    ('scaler' , StandardScaler(with_mean=False))
                ]
            )
            num_pipeline = Pipeline(
                steps = [
                    ('Imputer' , SimpleImputer(strategy='median')),
                    ('scaler' , StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('cat_features' , cat_pipeline , categorical_columns),
                    ('num_features' , num_pipeline , numerical_columns)
                ]
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self , train_path , test_path):
        try:
            logging.info('Read Train and Test data')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_obj = self.get_data_transformer_obj()

            logging.info('Seperating features and target variable')
            train_df_features = train_df.drop('math_score' , axis = 1)
            train_df_target = train_df['math_score']

            test_df_features = test_df.drop('math_score' , axis = 1)
            test_df_target = test_df['math_score']

            logging.info('Transforming Train and Test data')
            train_df_features_transformed = preprocessor_obj.fit_transform(train_df_features)
            test_df_features_transformed = preprocessor_obj.transform(test_df_features)

            train_data = np.column_stack((train_df_features_transformed , np.array(train_df_target)))
            test_data = np.column_stack((test_df_features_transformed , np.array(test_df_target)))

            logging.info('Saving the preprocessor object file')
            save_obj(preprocessor_obj , self.data_transformation_config.preprocessor_obj_file_path)

            logging.info('Data Transformation complete')
            return train_data , test_data , self.data_transformation_config.preprocessor_obj_file_path
            

        except Exception as e:
            raise CustomException(e,sys)
