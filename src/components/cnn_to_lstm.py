import torch
import sys
import torch.nn as nn
from src.exception import CustomException
from src.logger import logging
import statistics
import torch.nn.functional as F
import torchvision.models as models
from dataclasses import dataclass

@dataclass
class EncoderCNN(nn.Module):
    embed_size: int
    train_CNN: bool = False

    def __post_init__(self):
        super().__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        self.modules = list(self.inception.children())[:-1]  # exclude the last layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, self.embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        try:
            logging.info('Initializing encoder using inception')
            images = F.pad(images, (3, 3, 3, 3), mode='constant', value=0)
            images = images.unsqueeze(0)
            with torch.no_grad():
                features = self.inception(images)
            features = features.squeeze()
            features = self.relu(features)
            features = self.dropout(features)
            features = self.bn(features)
            logging.info('Initializing encoder over')
            return features
            
            
        except Exception as e:
            logging.error(f'Error occurred: {e}')
            raise CustomException(e,sys)
    def __hash__(self):
        # Implementation of hash function that returns the id of the object
        return id(self)
    
@dataclass
class DecoderRNN(nn.Module):
    embed_size: int
    hidden_size: int
    vocab_size: int
    num_layers: int

    def __post_init__(self):
        super().__init__()
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        try:
            logging.info('Initializing decoder using LSTM')
            embeddings = self.dropout(self.embed(captions))
            embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
            hiddens, _ = self.lstm(embeddings)
            outputs = self.linear(hiddens)
            return outputs
        except Exception as e:
            logging.error(f'Error occurred: {e}')
            raise CustomException(e,sys)
            
    def __hash__(self):
        # Implementation of hash function that returns the id of the object
        return id(self)
    
    
@dataclass
class CNNtoRNN(nn.Module):

    embed_size: int
    hidden_size: int
    vocab_size: int
    num_layers: int

    def __post_init__(self):
        super().__init__()
        self.encoderCNN = EncoderCNN(self.embed_size)
        self.decoderRNN = DecoderRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)

    # This is for the training purpose
    def forward(self, images, captions):
        try:
            logging.info('Starting overall Forward Propagation...')
            features = self.encoderCNN(images)
            outputs = self.decoderRNN(features, captions)
            logging.info('Forward propagation completed...')
            return outputs
        except Exception as e:
            logging.error(f'Error occurred: {e}')
            raise CustomException(e,sys)
            
    
    # This is for inference dataset
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        
        try:
            logging.info("Captioning images")
            with torch.no_grad():
                x = self.encoderCNN(image).unsqueeze(0)
                states = None
    
                for _ in range(max_length):
                    hiddens, states = self.decoderRNN.lstm(x, states)
                    output = self.decoderRNN.linear(hiddens.squeeze(0))
                    predicted = output.argmax(1)
                    result_caption.append(predicted.item())
                    x = self.decoderRNN.embed(predicted).unsqueeze(0)
    
                    if vocabulary.itos[predicted.item()] == "<EOS>":
                        break
            logging.info('Captioning images completed')
    
            return [vocabulary.itos[idx] for idx in result_caption]
        
        except Exception as e:
            logging.error(f'Error occurred: {e}')
            raise CustomException(e,sys)
            
    def __hash__(self):
        # Implementation of hash function that returns the id of the object
        return id(self)
    
    
