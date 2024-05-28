from transformer import *
from event_tokenizer import EventTokenizer
#from icecream import ic
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


# #model = Decoder(20_000, 1024, 16, 64, 4, 2, 0.1)
# vocab_size=20_000
# max_len=1024
# d_k=16
# d_model=64
# nb_heads=4
# nb_layers=2
# dropout_proba=0.1
# #model = Decoder(vocab_size, max_len, d_k, d_model, n_heads, n_layers, dropout_prob)

# model = Decoder(d_k, d_k, d_model, nb_heads, nb_layers, dropout_proba, max_len, vocab_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# ic(model.to(device))

# x = np.random.randint(0, 20_000, size=(8,512))
# x_t = torch.tensor(x).to(device)
# y = model(x_t)
# ic(y.shape)
# mask = np.ones((8,512))
# mask[:, 256:] = 0
# mask_t = torch.tensor(mask).to(device)

# y = model(x_t, mask_t)
# ic(y.shape)

tok = EventTokenizer(2)
vocab_size = tok.vocab_size

data = []

for filename in os.scandir("../BachTwoPartInventions/"):
    if filename.is_file():
        songs = tok.file_to_vector(filename, augment=True, return_mask=False)
        data += songs
    

max_length = max(len(song) for song in data)

def collate_fn(batch: list):
    max_length = max(len(song) for song in batch)
    masks = []
    songs = []
    for song in batch:
        padded = np.ones(max_length, dtype=int)
        padded[len(song):] = 0
        masks.append(padded)
        song += [0 for _ in range(max_length - len(song))]
        songs.append(song)
    return torch.tensor(np.array(songs)), torch.tensor(np.array(masks))
    
class BachDataset(Dataset):
    def __init__(self, data):
        self.data = data
        #self.songs, self.masks = self.process_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return data[idx]
    
train_loader = DataLoader(
    BachDataset(data),
    shuffle=True,
    batch_size=32,
    collate_fn=collate_fn
)

def train(model, criterion, optimizer, train_loader, epochs):
    train_losses = np.zeros(epochs)
    for i in range(epochs):
        model.train()
        starttime = datetime.now()
        train_loss = []
        for songs, masks in train_loader:
           songs = songs.to(device)
           masks = masks.to(device)

           optimizer.zero_grad()

           targets = songs.clone().detach()
           targets = torch.roll(targets, shifts=-1, dims=1)
           targets[:, -1] = 0

           outputs = model(songs, masks)

           loss = criterion(outputs.transpose(2,1), targets)

           loss.backward()
           optimizer.step()
           train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        train_losses[i] = train_loss

        elapsed = datetime.now() - starttime
        print(f"Epoch {i+1}/{epochs}, Train loss: {train_loss:.4f}, Duration: {elapsed}")
    return train_losses


# model = Decoder(d_k, d_k, d_model, nb_heads, nb_layers, dropout_proba, max_len, vocab_size)

model = Decoder(d_k=16, d_v=16, d_model=64, nb_heads=4, nb_layers=2, dropout_proba=0.1, max_len=max_length, vocab_size=vocab_size)
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters())

train_losses = train(model, criterion, optimizer, train_loader, epochs=1)

model.eval()

prompt = torch.tensor([[0, 62, 1041]])
mask = torch.tensor([[1,1,1]])

for _ in range(100):
    outputs = model(prompt, mask)
    prediction = torch.argmax(outputs[:,-1,:], axis=-1) #last one = predicted next
    prompt = torch.hstack((prompt, prediction.view(1, 1)))
    mask = torch.ones_like(prompt)

    if prediction == 1: #end token
        break

print(prompt)







