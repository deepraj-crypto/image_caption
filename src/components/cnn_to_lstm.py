import torch
import torch.nn as nn
from src.exception import CustomException
from src.logger import logging
import statistics
import torchvision.models as models
from dataclasses import dataclass

@dataclass
class EncoderCNN(nn.Module):
    embed_size: int
    train_CNN: bool = False

    def __post_init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        layers_to_keep = -1 if self.train_CNN else 7
        self.resnet = nn.Sequential(*list(self.resnet.children())[:layers_to_keep])
        self.linear = nn.Linear(self.resnet[-1][-1].bn2.num_features, self.embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        try:
            logging.info('Initializing encoder using resnet')
            features = self.resnet(images)
            features = self.dropout(self.relu(self.linear(features.transpose(1, 2)).transpose(1, 2)))
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
    
    

