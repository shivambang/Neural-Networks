import torch
import eval
import numpy as np
from torch import Tensor
from torch.autograd import Variable
from data import add_noise, normalize
from random import randint
def train(data, target, model, criterion, optimizer, epochs):
    model.train()
    clean_data = Variable(Tensor(data))
    target = Variable(Tensor(target))
    for e in range(epochs):
    
        # Forward pass: Compute predicted y by passing x to the model
        if e % 2 == 0:
            pred_y = model(clean_data)
        else:
            noisy_data = []
            for d in data:
                nd = add_noise(np.copy(d), cx=0.25, sigma=randint(1, 9)/10**randint(1, 3))
                noisy_data.append(normalize(nd))
            noisy_data = np.array(noisy_data)
            pred_y = model(Variable(Tensor(noisy_data)))

        # Compute loss
        loss = criterion(pred_y, target)
    
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(data, target, model):
    model.eval()
    target_values = generate_rescaled_inputs(target)
    predicted_values = generate_predictions(data, model)
    return predicted_values, eval.compute_error_metrics(target_values, predicted_values)

#convert numpy array to 16x16 binary image
def generate_rescaled_inputs(data):
    input_data = []
    
    for i in range(len(data)):
        p = data[i].reshape(16, 16)
        p[p <= 0.5] = 0
        p[p > 0.5] = 1
        input_data.append(p)
        
    return input_data

def generate_predictions(data, model):
    predicted_output = []
    with torch.no_grad():
        data = Variable(Tensor(data))
        pred = model(data).cpu().detach().numpy()
        for p in pred:
            p = p.reshape(16, 16)
            p[p <= 0.5] = 0
            p[p > 0.5] = 1
            predicted_output.append(p)
    
    return predicted_output