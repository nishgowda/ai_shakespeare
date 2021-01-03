#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
from helpers import *
from model import *
from generate import *
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_gpu = True if device == "cuda" else False

if is_gpu:
    print("GPU detected, training with CUDA")
else:
    print("No GPU detected, training with CPU")

file_name = 'data/shakespeare.txt'
file, file_len = read_file(file_name)

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if is_gpu:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

chunk_len = 200
batch_size = 200
hidden_size = 100
n_layers = 2
learning_rate = 0.01
n_epochs = 2000
print_every = 100
model = "gru"

decoder = RNN(
    n_characters,
    hidden_size,
    n_characters,
    model=model,
    n_layers=n_layers,
)

decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
start = time.time()
all_losses = []
loss_avg = 0

def train(inp, target):
    hidden = decoder.init_hidden(batch_size)
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data / chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(file_name))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

# Initialize models and start training
try:
    print("Training for %d epochs..." % n_epochs)
    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = train(*random_training_set(chunk_len, batch_size))
        loss_avg += loss

        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
            print(generate_text(decoder, 'Wh', 100, cuda=is_gpu), '\n')

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()
