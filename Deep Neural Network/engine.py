import eval
from torch import Tensor
from torch.autograd import Variable

def train(data, target, model, criterion, optimizer, epochs):
    model.train()
    data = Variable(Tensor([data]))
    target = Variable(Tensor([target]))
    for _ in range(epochs):
    
        # Forward pass: Compute predicted y by passing x to the model
        pred_y = model(data)
    
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
    
    for i in range(len(data)):
        p = model(Variable(Tensor([data[i]]))).cpu().detach().numpy()
        p = p.reshape(16, 16)
        p[p <= 0.5] = 0
        p[p > 0.5] = 1
        predicted_output.append(p)
    
    return predicted_output