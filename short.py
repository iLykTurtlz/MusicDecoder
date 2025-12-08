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
import csv

BATCH_SIZE=2
EPOCHS=200

# #model = Decoder(20_000, 1024, 16, 64, 4, 2, 0.1)
# vocab_size=20_000
# max_len=1024
# d_k=16
# d_model=64
# nb_heads=4
# nb_layers=2
# dropout_proba=0.1
F

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

tok = EventTokenizer(5)
vocab_size = tok.vocab_size

data = []

for filename in os.scandir("/datasets/pjarski/BachTwoPartInventions/"):
    if filename.is_file():
        songs = tok.file_to_vector(filename, augment=True, return_mask=False)
        data += songs

for filename in os.scandir("/datasets/pjarski/BachThreePartInventions/"):
    if filename.is_file():
        songs = tok.file_to_vector(filename, augment=True, return_mask=False)
        data += songs

for filename in os.scandir("/datasets/pjarski/BachFugues/"):
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
    batch_size=BATCH_SIZE,
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
        #this didn't decrease the slowdown of epochs
        #torch.cuda.empty_cache()
        #gc.collect()
    return train_losses


# model = Decoder(d_k, d_k, d_model, nb_heads, nb_layers, dropout_proba, max_len, vocab_size)

init_args = {
   "d_k":256, 
   "d_v":256, 
   "d_model":256, 
   "nb_heads":8, 
   "nb_layers":2, 
   "dropout_proba":0.1, 
   "max_len":max_length, 
   "vocab_size":vocab_size,
}


model = Decoder(**init_args)





#model = Decoder(d_k=16, d_v=16, d_model=64, nb_heads=4, nb_layers=2, dropout_proba=0.1, max_len=max_length, vocab_size=vocab_size)
model.to(device)

criterion = nn.CrossEntropyLoss()#(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters())

train_losses = train(model, criterion, optimizer, train_loader, epochs=EPOCHS)

model.eval()



current_time = datetime.now()
datetime_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

save_path = f"/datasets/pjarski/model_with_args_{datetime_str}.pth"

# Save the state dictionary and initialization arguments
torch.save({
    'model_state_dict': model.state_dict(),
    'init_args': init_args
}, save_path)

'''
Load later:

# Define the path to load the model
load_path = "model_with_args.pth"

# Load the checkpoint
checkpoint = torch.load(load_path)

# Extract the initialization arguments and state dictionary
init_args = checkpoint['init_args']
model_state_dict = checkpoint['model_state_dict']

# Reinitialize the model with the saved arguments
model = MyModel(**init_args)

# Load the saved state dictionary into the model
model.load_state_dict(model_state_dict)

'''

prompt = torch.tensor([[0]]).to(device)
mask = torch.tensor([[1]]).to(device)

for _ in range(3000):
    outputs = model(prompt, mask)
    prediction = torch.argmax(outputs[:,-1,:], axis=-1) #last one = predicted next
    prompt = torch.hstack((prompt, prediction.view(1, 1))).to(device)
    mask = torch.ones_like(prompt).to(device)

    if prediction == 1: #end token
        break

prompt = np.array(prompt.to('cpu'))
tok.vector_to_midi(prompt, f"/datasets/pjarski/test_gen_{datetime_str}.mid")

#write losses to csv
loss_file = f"/datasets/pjarski/losses_{datetime_str}.csv"
with open(loss_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    for loss_value in train_losses:
        writer.writerow([loss_value])

print(f"Loss values have been written to {loss_file}")





