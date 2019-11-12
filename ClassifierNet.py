import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.nn import functional as F

parser = argparse.ArgumentParser(description='ClassifierNet')
parser.addArgument('--batchsize', type=int, default=100, metavar='b',
                    help = 'training and testing batch size (default=100)')
args = parser.parse_args()

#gets training and testing data for MNIST dataset
trainData = datasets.MNIST('data/', train = True, transform = transforms.ToTensor(), download=True)
testData = datasets.MNIST('data/', train = False, transform = transforms.ToTensor(), download=True)

#constructs loaders from datasets
trainLoader = torch.utils.data.DataLoader(trainData, batch_size=args.batchsize, shuffle=True)
testLoader = torch.utils.data.DataLoader(testData, batch_size=args.batchsize, shuffle=True)



'''
Convolutional Neural Network that transforms an 1x28x28 image to a 128x2x2 feature map
'''
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #(1,28,28) -> (16,24,24)
        self.conv1 = nn.Conv2d(1, 16, 5)
        
        #(16,12,12) -> (32,10,10)
        self.conv2 = nn.Conv2d(16, 32, 3)
        
        #(32,5,5) -> (64,3,3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        
        #(64,3,3) -> (64,3,3)
        self.dropout = nn.Dropout2d(.2)
        
        #(64,3,3) -> (128,2,2)
        self.conv4 = nn.Conv2d(64, 128, 2)
        
    def forward(self, x):
        #(1,28,28) -> (16,24,24)
        x = F.leaky_relu(self.conv1(x))
        
        #(16,24,24) -> (16,12,12)
        x = F.max_pool2d(x, (2,2))
        
        #(16,12,12) -> (32,10,10)
        x = F.leaky_relu(self.conv2(x))
        
        #(32,10,10) -> (32,5,5)
        x = F.max_pool2d(x, (2,2))
        
        #(32,5,5) -> (64,3,3)
        x = F.leaky_relu(self.conv3(x))
        
        #(64,3,3) -> (64,3,3)
        x = self.dropout(x)
        
        #(64,3,3) -> (128,2,2)
        x = F.leaky_relu(self.conv4(x))
        
        return x        



'''
Fully-Connected network that transforms 512 inputs to 10 softmaxed outputs
'''
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        #512 -> 128
        self.linear1 = nn.Linear(512, 128)
        
        #128 -> 10
        self.linear2 = nn.Linear(128, 10)
    def forward(self, x):
        #512 -> 128
        x = F.leaky_relu(self.linear1(x))
        
        #128 -> 10
        x = F.softmax(self.linear2(x), dim=1)
        
        return x



'''
MNIST classifier composed of ConvNet and LinearNet
'''
class ClassifierNet(nn.Module):
    def __init__(self):
        super(ClassifierNet, self).__init__()
        
        #(1,28,28) -> (128,2,2)
        self.convNet = ConvNet()
        
        #512 -> 10
        self.linearNet = LinearNet()
    def forward(self, x):
        x = self.convNet(x)
        
        x = x.view(-1, 512)
        
        x = self.linearNet(x)
        
        return x
model = ClassifierNet()



'''
Optimizer constructors

https://pytorch.org/docs/stable/optim.html
'''

#Constructs a stochastic gradient descent optimizer with a learning rate of 1e-3 (0.001) on our model
optimizer1 = optim.SGD(model.parameters(), lr=1e-3)

#Constructs an Adam optimizer with a learning rate of 1e-3 on convolutional layers, 1e-2 on the linear layers of our model
optimizer2 = optim.Adam([{'params': model.convNet.parameters()},
                         {'params': model.linearNet.parameters(), 'lr': 1e-2}], lr = 1e-3)
optimizer = optimizer2



'''
Makes a training method, explaining the different components
'''

def train():
    #initializes an accumulator to track the total loss over a training epoch
    trainingLoss = 0
    
    #sets up a loop through the entire dataset
    #index denotes the index of the batch being processed
    #data is the batch of pictures to be processed through the network
    #target is the corresponding batch of actual classification values we want to model
    for index, (data, target) in enumerate(trainLoader):
        #zeros out the gradient on the optimizer
        #this prepares the optimizer to record weight updates for the new batch
        optimizer.zero_grad()
        
        #runs the model on the data, storing the output
        predictions = model(data)
        
        #calculates the loss/cost function for the batch, modeling the error on the current batch
        batchLoss = F.cross_entropy(predictions, target)
        
        #backpropagates the loss through the network to determine how to update the weights of the network
        batchLoss.backward()
        
        #adds the current batch's loss to the total loss for the training epoch
        #.item() gets the actual numerical value for the batch's loss
        trainingLoss += batchLoss.item()
        
        #updates the network's weights using the optimizer
        optimizer.step()
        
        #prints the average loss for a single image from the current batch every ten batches
        if index % 10 == 0:
            print(f"Batch Loss: {batchLoss.item() / len(data)}")
            
    #prints the total average loss for the training epoch
    print(f"Average Loss for Epoch: {trainingLoss / len(trainData)}")



'''
Makes a testing method, explaining the different components
'''

def test():
    #initializes an accumulator to track the total loss over a testing data
    testingLoss = 0
    
    #initializes an accumulator to track how many images are processed
    total = 0
    
    #initializes an accumulator to track how many images' values were correctly predicted
    correct = 0
    
    #specifies that gradients should not be tracked through these observations
    #this speeds up computation time for processes irrelevant to the training of the network
    with torch.no_grad():
        #sets up a loop through the entire dataset
        #index denotes the index of the batch being processed
        #data is the batch of pictures to be processed through the network
        #target is the corresponding batch of actual classification values we want to model
        for index, (data, target) in enumerate(testLoader):

            #runs the model on the data, storing the output
            predictions = model(data)

            #calculates the loss/cost function for the batch and adds it to the total loss
            testingLoss += F.cross_entropy(predictions, target).item()
            
            #gets the most likely classification for each image
            _, predictedValues = torch.max(predictions, 1)

            #adds the number of images in the current batch to the total
            total += target.size(0)
            
            #adds the number of correctly classified images to the total correct
            correct += (predictedValues == target).sum().item()
            
    print(f"Test set loss: {testingLoss}")
    print(f"Accuracy: {correct/total}")



#runs the model for 1 epoch
for epoch in range(1, 2):
    print(f"Epoch Number {epoch}")
    train()
    test()
    #saves the weights for later use
    torch.save({'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()
               }, 'new_weights.h5')