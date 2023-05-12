import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
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
            for col in df.columns:
                numerical_column.append(col)
                
            numerical_column = numerical_column[1:(len(numerical_column)-1)]
                  
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
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
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data successfully...")
            logging.info("Obtaining preprocessing object...")
    
            preprocessing_obj_train = self.get_data_transformer_obj(train_path)
            logging.info("msg2")
            preprocessing_obj_test = self.get_data_transformer_obj(test_path)
            logging.info("msg3")
            
            target_column_name = "yield"
            
            x_train = train_df.drop(["id", "yield"], axis=1)
            y_train = train_df[target_column_name]
            x_test = test_df.drop(["id", "yield"], axis=1)
            y_test = test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on training dataset and testing dataset.")
            
            x_arr = preprocessing_obj_train.fit_transform(x_train)
            y_arr = np.array(y_train)
            x_test_arr = preprocessing_obj_test.fit_transform(x_test)
            y_test_arr = np.array(y_test)
            
            train_arr = np.c_[x_arr,y_arr]
            test_arr = np.c_[x_test_arr,y_test_arr]
            
            logging.info(f"Saved preprocessing object...")
            
            save_object(
                file_path=self.data_transformer_config.preprocessor_ob_file_path,
                obj = preprocessing_obj_train
            )
            
            logging.info("msg")
            
            return(
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_ob_file_path
            )
            
        except Exception as e:
            CustomException(e, sys)