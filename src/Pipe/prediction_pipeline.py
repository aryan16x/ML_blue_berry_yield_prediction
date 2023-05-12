import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class predict_pipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = "artifacts\preprocessor.pkl"
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)
    
class custom_data:
    def __init__(self,
                clonesize:float,
                honeybee:float,
                bumbles:float,
                andrena:float,
                osmia:float,
                MaxOfUpperTRange:float,
                MinOfUpperTRange:float,
                AverageOfUpperTRange:float,
                MaxOfLowerTRange:float,
                MinOfLowerTRange:float,
                AverageOfLowerTRange:float,
                RainingDays:float,
                AverageRainingDays:float,
                fruitset:float,
                fruitmass:float,
                seeds:float
                ):
        self.clonesize = clonesize
        self.honeybee = honeybee
        self.bumbles = bumbles
        self.andrena = andrena
        self.osmia = osmia
        self.MaxOfUpperTRange = MaxOfUpperTRange
        self.MinOfUpperTRange = MinOfUpperTRange
        self.AverageOfUpperTRange = AverageOfUpperTRange
        self.MaxOfLowerTRange = MaxOfLowerTRange
        self.MinOfLowerTRange = MinOfLowerTRange
        self.AverageOfLowerTRange = AverageOfLowerTRange
        self.RainingDays = RainingDays
        self.AverageRainingDays = AverageRainingDays
        self.fruitset = fruitset
        self.fruitmass = fruitmass
        self.seeds = seeds
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "clonesize" : [self.clonesize],
                "honeybee" : [self.honeybee], 
                "bumbles" : [self.bumbles],
                "andrena" : [self.andrena],
                "osmia" : [self.osmia],
                "MaxOfUpperTRange" : [self.MaxOfUpperTRange],
                "MinOfUpperTRange" : [self.MinOfUpperTRange],
                "AverageOfUpperTRange" : [self.AverageOfUpperTRange],
                "MaxOfLowerTRange" : [self.MaxOfLowerTRange],
                "MinOfLowerTRange" : [self.MinOfLowerTRange],
                "AverageOfLowerTRange" : [self.AverageOfLowerTRange],
                "RainingDays" : [self.RainingDays],
                "AverageRainingDays" : [self.AverageRainingDays],
                "fruitset" : [self.fruitset],
                "fruitmass" : [self.fruitmass],
                "seeds" : [self.seeds]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
        