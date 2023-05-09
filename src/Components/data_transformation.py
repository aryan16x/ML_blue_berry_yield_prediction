import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleInputer
from sklearn.pipeline import Pipeline
from skearn.preprocessing import OneHotEncoder,StandardScaler
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class data_transformer_config:
    preprocessor_ob_file_path = os.path.join('artifacts',"preprocessor.pkl")
    
class data_transformer:
    def __init__(self):
        self.data_transformer_config = data_transformer_config()
        
    def get_data_transformer_obj(self,df_path):
        try:
            df = pd.read_csv(df_path)
            numerical_column = []
            for col in x.columns:
                numerical_column.append(col)
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleInputer(strategy="median")),
                    ("scalar", StandardScaler())
                ]
            )
            
            logging.info("Numerical columns standard scaling completed...")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_column)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data successfully...")
            logging.info("Obtaining preprocessing object...")
            
            preprocessing_obj_train = self.get_data_transformer_obj(train_path)
            
            target_column_name = "yield"
            
            x = train_df.drop(["id", target_column_name], axis=1)
            y = train_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on training dataset and testing dataset.")
            
            x_arr = preprocessing_obj_train.fit_transform(x)
            
            train_arr = np.c_[x_arr, np.array(y)]
            
            logging_info(f"Saved preprocessing object...")
            
            save_object(
                file_path=self.data_transformer_config.preprocessor_ob_file_path,
                obj = preprocessing_obj_train
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_ob_file_path
            )
            
        except Exception as e:
            CustomException(e, sys)