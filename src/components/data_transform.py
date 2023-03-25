import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import CaptionDataIngestion

#df = pd.read_csv()


@dataclass
class Data_transformation:
    caption_path: str
    col_num: int # to drop a specific column

    def initiate_data_transformation(self):
        logging.info('data transformation initiated')
        try:
            df = pd.read_csv(self.caption_path)
            if df.columns[self.col_num]:
                df = df.drop(df.columns[[self.col_num]], axis=1)
                df.head()
                df.to_csv(self.caption_path, sep=',', index=False)
                logging.info('Data transformation completed')
                
                return (
                    self.caption_path
                    )
            else:
                logging.warning(f'Column {self.col_num} not found in file')
        except Exception as e:
            logging.error(f'Error occurred: {e}')

if __name__ == '__main__':
    caption_data=CaptionDataIngestion()
    data_transform = Data_transformation(caption_data.initiate_data_ingestion(), 1)
    data_transform.initiate_data_transformation()