import os
import sys
from dataclasses import dataclass
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.exception import CustomException
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
            x_train,x_test,y_train,y_test = train_test_split(train_arr[:,:-1], train_arr[:,-1], test_size=0.2, random_state=42)
            # models = {"Linear Regressin" : LinearRegressin()}
            
            # model_report: dict = evaluate_model(
            #     x_train = x_train,
            #     y_train = y_train,
            #     x_test = x_test,
            #     y_test = y_test,
            #     models = models
            # )
            
            lr = LinearRegression()
            lr.fit(x_train,y_train)
            y_pred = lr.predict(x_test)
            
            # r2_score = r2_score(y_test,y_pred)
            
            x = lr.score(x_train, y_train)
            
            logging.info("Model saving is started...")
            
            pickle.dump(lr, open('artifacts/model.pkl','wb'))
            
            logging.info("ML model saved...")
            
            return x
        
        except Exception as e:
            CustomException(e,sys)