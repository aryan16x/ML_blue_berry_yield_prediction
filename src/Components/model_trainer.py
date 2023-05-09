import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearModel
from sklearn.metrics import r2_score

from src.exceptions import CustomException
from src.logger import logging

@dataclass
class model_trainer_config:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class model_trainer:
    def __init__(self):
        self.model_trainer_config = model_trainer_config()
        
    def initiate_model_trainer(self,train_arr,test_arr,preprocessor_path):
        try:
            logging.info("Splitting training and testing data...")
            x_train,y_train,x_test,y_test = sklearn.model_selection.train_test_split(train_arr[:,:-1], train_arr[:,-1], test_size=0.2, random_state=42)
            # models = {"Linear Regressin" : LinearRegressin()}
            
            # model_report: dict = evaluate_model(
            #     x_train = x_train,
            #     y_train = y_train,
            #     x_test = x_test,
            #     y_test = y_test,
            #     models = models
            # )
            
            # save_object(
            #     file_path = self.model_trainer_config.trained_model_file_path,
            #     obj = LinearRegression
            # )
            
            lr = LinearRegression()
            lr.fit(x,y)
            y_pred = lr.predict(x_test)
            
            r2_score = r2_score(y_test,predicted)
            
            return r2_score
        
        except Exception as e:
            CustomException(e,sys)