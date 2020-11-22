import os
import torch.optim

from data_loader import TextDS
from models import FastText
from plotting import *
from training import train_model
import __settings__

num_epochs = 25
num_hidden = 4096

__settings__.split = (0.95, 0.025)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = TextDS(os.path.join('data', 'labelled_movie_reviews.csv'), dev=dev)

model = FastText(len(dataset.token_to_id)+2, num_hidden, len(dataset.token_to_id)+2).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses, accuracies = train_model(dataset, model, optimizer, num_epochs)
torch.save(model, os.path.join('saved_models', 'word_embeddings.pth'))

print_accuracies(accuracies)
plot_losses(losses)
