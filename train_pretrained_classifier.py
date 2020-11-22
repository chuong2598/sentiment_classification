import os
import torch.optim

from data_loader import LabelledTextDS
from models import FastText
from plotting import *
from training import train_model

num_epochs = 15
num_hidden = 4096  # Number of hidden neurons in model

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = LabelledTextDS(os.path.join('data', 'labelled_movie_reviews.csv'), dev=dev)

embeddings = torch.load(os.path.join('saved_models', 'word_embeddings.pth')).embeddings.weight.data
model = FastText(len(dataset.token_to_id)+2, num_hidden, len(dataset.class_to_id), word_embeddings=embeddings).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses, accuracies = train_model(dataset, model, optimizer, num_epochs)
torch.save(model, os.path.join('saved_models', 'classifier.pth'))

print('')
print_accuracies(accuracies)
plot_losses(losses)
