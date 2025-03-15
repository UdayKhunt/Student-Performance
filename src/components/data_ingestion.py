from dataclasses import dataclass
import os,sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_path:str = os.path.join('artifacts' , 'train.csv')
    test_path:str = os.path.join('artifacts' , 'test.csv')
    raw_data_path:str = os.path.join('artifacts' , 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Started')       
        try:
            #make directory if not present, if present:ignore
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_path) , exist_ok=True)
            df = pd.read_csv('notebooks/data/stud.csv')
            
            logging.info('Raw data stored')       

            df.to_csv(self.data_ingestion_config.raw_data_path,header=True,index=False)

            logging.info('Train Tests split initiated')       
            train_set , test_set = train_test_split(df , test_size=.2)
            
            logging.info('Train and Test data stored')       
            train_set.to_csv(self.data_ingestion_config.train_path , header=True , index=False)
            test_set.to_csv(self.data_ingestion_config.test_path , header=True , index=False)

            logging.info('Data Ingestion completed')       
            return (self.data_ingestion_config.train_path , self.data_ingestion_config.test_path)

            
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_path , test_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_data , test_data , _  = data_transformation.initiate_data_transformation(train_path , test_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_data , test_data))

            




