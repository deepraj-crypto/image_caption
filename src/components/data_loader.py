'''
 reference: https://youtu.be/8P8PBxj9IEw 
            https://youtu.be/y7JCHwp7qFs
            https://youtu.be/b9qurH0_g5s
'''

import os
import sys
import pandas as pd 
import spacy 
import torch
from collections import Counter
from src.exception import CustomException
from src.logger import logging
from src.components.data_transform import Data_transformation
from typing import Optional, List, Callable
from dataclasses import dataclass, field
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import torchvision.transforms as transforms

NUM_WORKERS = os.cpu_count()


nlp = spacy.load("en_core_web_sm")

@dataclass
class Vocabulary:
    '''
        Creating Vocabulary
    '''
    freq_threshold: int
    itos: dict = field(default_factory=lambda: {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"})
    stoi: dict = field(default_factory=lambda: {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3})

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in nlp(text)]

    def build_vocabulary(self, sentence_list):
        try:
            logging.info('building vocabulary')
            word_counts = Counter()
            for sentence in sentence_list:
                if isinstance(sentence, float):
                    sentence = str(sentence)
                word_counts.update(self.tokenizer_eng(sentence))
    
            for word, count in word_counts.items():
                if count >= self.freq_threshold and word not in self.stoi:
                    idx = len(self.stoi)
                    self.stoi[word] = idx
                    self.itos[idx] = word
            logging.info('completed building vocabulary')
        except Exception as e:
            logging.error(f'Error occurred: {e}')
            raise CustomException(e,sys)

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]



@dataclass
class create_dataloader(Dataset):
    base_dir: str
    caption_data: str
    transform: Optional[Callable] = None
    freq_threshold: int = 5
    df: Optional[pd.DataFrame] = None
    img: Optional[pd.Series] = None
    comments: Optional[pd.Series] = None
    vocab: Optional[Vocabulary] = None

    def __post_init__(self):
        self.df = pd.read_csv(self.caption_data)
        self.img = self.df["image_name"]
        self.comments = self.df['comment']
        self.vocab = Vocabulary(self.freq_threshold)
        self.vocab.build_vocabulary(self.comments.tolist())
        
    def __len__(self):
        return len(self.df)
    
    # now get a single image with corresponding caption/comment
    def _getitem__(self, index):
        comment=self.comments[index]
        img_id=self.imgs[index]
        img=Image.open(os.path.join(self.base_dir,img_id)).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(comment)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)
        
        
@dataclass
class MyCollate:
    pad_idx: int

    def __call__(self, batch: List[tuple]) -> tuple:
        imgs = torch.cat([item[0].unsqueeze(0) for item in batch], dim=0)
        targets = pad_sequence([item[1] for item in batch], batch_first=False, padding_value=self.pad_idx)
        return imgs, targets


def get_loader(
    base_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=NUM_WORKERS,
    shuffle=True,
    pin_memory=True,
):
    dataset = create_dataloader(base_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


def run_main():
    
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    loader, dataset = get_loader(
        "artifacts/flickr_img", "artifacts/comment.csv", transform=transform
    )

    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)
        
if __name__=='__main__':
    run_main()
    logging.info('Data loader test successfull')
    
        
        
        
        