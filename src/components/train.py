'''
Reference: https://www.learnpytorch.io/
'''

import os
import sys
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from src.exception import CustomException
from src.logger import logging
from src.utils import save_checkpoint, load_checkpoint, print_examples, TestExample
from src.components.data_loader import get_loader
from src.components.cnn_to_lstm import CNNtoRNN

def train():
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    try:

        train_loader, dataset = get_loader(
            base_folder="artifacts/flickr_img",
            annotation_file="artifacts/comment.csv",
            transform=transform,
            num_workers=os.cpu_count(),
        )
        logging.info('Tranforming data completed')
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise CustomException(e,sys)
    
# =============================================================================
#     Here we can create a test loader too by manually splitting the data and then its corresponding captions too.
#     But we will use some manual test cases.
# =============================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = False
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 5
    learning_rate = 3e-4
    num_epochs = 100

    try:
        logging.info('Initializing model')
        model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        logging.info('Initializing model completed')
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise CustomException(e,sys)

    try:
        logging.info('Finetuning cnn model')
        for name, param in model.encoderCNN.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = train_CNN
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise CustomException(e,sys)

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        # Uncomment the line below to see a couple of test cases
        test_examples = [TestExample(image_path="foot.jpg", correct_caption="Boys playing football in mud in the slums")]

        print_examples(model, device, dataset, test_examples)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
            
            
            
# =============================================================================
#             
# def train():
#     transform = transforms.Compose([
#         transforms.Resize((356, 356)),
#         transforms.RandomCrop((299, 299)),
#         transforms.ToTensor(),
#     ])
# 
#     dataset = datasets.ImageFolder('path/to/training/dataset', transform=transform)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
# 
#     # Calculate mean and standard deviation of training dataset
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     for i, (images, _) in enumerate(dataloader):
#         batch_samples = images.size(0)
#         images = images.view(batch_samples, images.size(1), -1)
#         mean += images.mean(2).sum(0)
#         std += images.std(2).sum(0)
#     mean /= len(dataset)
#     std /= len(dataset)
# 
#     # Define transformations with new mean and standard deviation values
#     transform = transforms.Compose([
#         transforms.Resize((356, 356)),
#         transforms.RandomCrop((299, 299)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#     ])
# 
#     try:
#         train_loader, dataset = get_loader(
#             base_folder="artifacts/flickr_img",
#             annotation_file="artifacts/comment.csv",
#             transform=transform,
#             num_workers=os.cpu_count(),
#         )
#         logging.info('Transforming data completed')
#     except Exception as e:
#         logging.error(f'Error occurred: {e}')
#         raise CustomException(e,sys)
# 
#     # rest of the code ...
# }
# 
# =============================================================================


if __name__ == "__main__":
    train()