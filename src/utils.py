import sys
import torch
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import torchvision.transforms as transforms
from PIL import Image
from dataclasses import dataclass
from typing import List



@dataclass
class TestExample:
    image_path: str
    correct_caption: str

def print_examples(model, device, dataset, test_examples: List[TestExample]):
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# =============================================================================
#   The training this way takes time because of my system specs, so I will pass hand writen captions.
#     df = pd.read_csv(csv_file_path)
#     test_examples = [
#         TestExample(os.path.join(image_folder_path, row["filename"]), row["caption"])
#         for _, row in df.iterrows()
# =============================================================================
# =============================================================================
#     ]
# 
#     for example in test_examples:
#         test_img = transform(Image.open(example.image_path).convert("RGB")).unsqueeze(0)
#         output = model.caption_image(test_img.to(device), dataset.vocab)
#         predicted_caption = " ".join(output)
#         print(f"Example {test_examples.index(example) + 1} CORRECT: {example.correct_caption}")
#         print(f"Example {test_examples.index(example) + 1} OUTPUT: {predicted_caption}")
# =============================================================================
        
    try:
        
        logging.info('Testing the correctness of an example')
        for example in test_examples:
            test_img = transform(Image.open(example.image_path).convert("RGB")).unsqueeze(0)
            output = model.caption_image(test_img.to(device), dataset.vocab)
            predicted_caption = " ".join(output)
            print(f"Example {test_examples.index(example) + 1} CORRECT: {example.correct_caption}")
            print(f"Example {test_examples.index(example) + 1} OUTPUT: {predicted_caption}")
            logging.info(f"Example {test_examples.index(example) + 1} CORRECT: {example.correct_caption}")
            logging.info(f"Example {test_examples.index(example) + 1} OUTPUT: {predicted_caption}")

    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise CustomException(e,sys)
        
        
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    try:
        logging.info('Saving Checkpoint...')
        print("=> Saving checkpoint")
        torch.save(state, filename)
        logging.info('Checkpoint saved')
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise CustomException(e,sys)


def load_checkpoint(checkpoint, model, optimizer):
    try:
        logging.info('Loading checkpoint...')
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        logging.info('CheckPoint Loaded>>')
        return step
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise CustomException(e,sys)
