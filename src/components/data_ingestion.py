'''
 If we were to import data from any big data sources like mongodb, etc then we
 need to get the input from that source and save it in out directory for data
 transformation. This is what we are doing here ->
'''


import os
import sys
import shutil
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd


@dataclass
class CaptionDataIngestionConfig:
    comment_path: str=os.path.join('artifacts',"comment.csv")

class CaptionDataIngestion:
    def __init__(self):
        self.ingestion_config=CaptionDataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info('Data ingestion started')
        try:
            df=pd.read_csv('results.csv')
            logging.info('Reading the comments/captions of the images')
            os.makedirs(os.path.dirname(self.ingestion_config.comment_path),exist_ok=True)

            df.to_csv(self.ingestion_config.comment_path,index=False,header=True)

            logging.info("Ingestion completed")
            
            return(
                self.ingestion_config.comment_path
                )
        
        except Exception as e:
            raise CustomException(e,sys)
            
            
            
@dataclass
class image_data_ingestion:
    image_path: str
    
    def initiate_img_data_ingestion(self, folder_path:str):
        try:
            os.makedirs(folder_path, exist_ok=True)
            logging.info('Image data ingestion')
            for file in os.listdir(self.image_path):
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    src_path = os.path.join(self.image_path, file)
                    dst_path = os.path.join(folder_path, file)
                    shutil.move(src_path, dst_path)
            logging.info('image ingestion completed')
            return(
                folder_path
                )
        except Exception as e:
            raise CustomException(e,sys)

        
        
if __name__=='__main__':
    image_folder = image_data_ingestion(image_path="flickr30k-images")
    image_folder.initiate_img_data_ingestion(folder_path="artifacts/flickr_img")
    
    obj=CaptionDataIngestion()
    obj.initiate_data_ingestion()
