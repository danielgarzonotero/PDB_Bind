import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from torch.utils.data import DataLoader, Dataset


# functional sections can be made using '#%% Title' instead of '# --- Title ---'
#%% Model

class Linear_Net(nn.Module): 
    
    def __init__(self,hidden_size1=5): # moved hidden size to make it an input
        super(Linear_Net, self).__init__() 
        input_size = 2048
        output_size = 1

        self.fl1 = nn.Linear(input_size, hidden_size1)
        self.fl2 = nn.Linear(hidden_size1, output_size) 
        
    def forward(self, x):
        out = self.fl1(x)
        out = F.relu(out) 
        out = self.fl2(out)
        
        return out 

#%% Train routine

def train(model, device, train_dataloader, optim): 
    model.train()

    loss_func = torch.nn.L1Loss(reduction='sum') 
    loss_collect = 0

    # fp, y redefined as input_vectors, targets for clarity
    for b_i, (input_vectors, targets) in enumerate(train_dataloader):

        input_vectors, targets = input_vectors.to(device), targets.to(device)

        optim.zero_grad() 
        pred_prob = model(input_vectors.float())

        loss = loss_func(pred_prob, targets.view(-1,1))
        loss.backward() 

        optim.step() 
        loss_collect += loss.item() 
    

        if b_i % 10 == 0:
            print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.1f}'.format(
                epoch, b_i * len(input_vectors), len(train_dataloader.dataset),
                100 * b_i * len(input_vectors) / len(train_dataloader.dataset),
                loss.item()))
    loss_collect /= len(train_dataloader.dataset)

    return loss_collect

#%% Validation Routine

def validate(model, device, val_dataloader, epoch): 

    model.eval()
    loss_collect = 0 
    loss_func = torch.nn.L1Loss(reduction='sum') 
    
    with torch.no_grad(): 
        for input_vectors, targets in val_dataloader:
            input_vectors, targets = input_vectors.to(device), targets.to(device) 
            pred_prob = model(input_vectors)
            loss_collect += loss_func(pred_prob, targets.view(-1,1)).item()
  
    loss_collect /= len(val_dataloader.dataset)
    
    print('\nTest dataset: Overall Loss: {:.1f},  ({:.2f}%)\n'.format(
        len(val_dataloader.dataset)*loss_collect,loss_collect))

    return loss_collect

#%% Dataset

class PDB(Dataset):
    def __init__(self,path):

        self.df = pd.read_csv(path)

        self.input_vectors= self.df[self.df.columns[0:-1]].values
        
        self.targets = self.df[self.df.columns[-1]].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index): 
        input_vector = self.input_vectors[index]
        target = self.targets[index]
        
        return torch.tensor(input_vector, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

#%% Prediction - moved, define functions and classes before code sections

def predict(model, device, dataloader):

    model.eval()

    input_vectors_all = []
    targets_all = []
    pred_prob_all = []

    # Remove gradients:
    with torch.no_grad():

        # Looping over the dataloader allows us to pull out input/output data:
        for input_vector, target in dataloader:

            # Assign X and y to the appropriate device:
            input_vector, target = input_vector.to(device), target.to(device)

            # Make a prediction:
            pred_prob = model(input_vector)

            input_vectors_all.append(input_vector)
            targets_all.append(target)
            pred_prob_all.append(pred_prob)
    return (
        torch.concat(input_vectors_all), 
        torch.concat(targets_all), 
        torch.concat(pred_prob_all).view(-1))

#%% Data Loader

data_set = PDB('pdbind_full_fp2.csv') 

bat_size = 100

# important to use split for test data and validation data
size_train = int(len(data_set) * 0.8)
size_val = len(data_set) - size_train
train_set, val_set = torch.utils.data.random_split(data_set, 
    [size_train, size_val], generator=torch.Generator().manual_seed(0))

train_loader = DataLoader(dataset = train_set,batch_size=bat_size,shuffle=True)

val_loader = DataLoader(dataset = val_set,batch_size=bat_size,shuffle=True)

#%% Run training loop

learning_rate = 0.001
torch.manual_seed(0)
device = torch.device("cpu")

model = Linear_Net() 
optimizer = optim.Adam(model.parameters(), lr = learning_rate) 

losses_train = []
losses_val = []

for epoch in range(1, 31): 
    train_loss = train(model, device, train_loader, optimizer)
    losses_train.append(train_loss)

    val_loss = validate(model, device, val_loader, epoch)
    losses_val.append(val_loss)

plt.plot(losses_train, label ='train losses')
plt.legend()
plt.xlabel('time')
plt.ylabel('train losses')

plt.plot(losses_val, label ='validation losses')
plt.legend()
plt.xlabel('time')
plt.ylabel('losses')

plt.show()

#%% Model Statistics

input_all, target_all, pred_prob_all = predict(model, device, val_loader)

r2 = r2_score(target_all, pred_prob_all)
mae = mean_absolute_error(target_all, pred_prob_all)
rmse = mean_squared_error(target_all, pred_prob_all, squared=False)

# only a few digits are relevant
print("R2 Score: {:.4f}".format(r2))
print("MAE: {:.4f}".format(mae))
print("RMSE: {:.4f}".format(rmse))
#%% Prediction Plot

plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(target_all, pred_prob_all, alpha=0.3)
plt.plot([min(target_all), max(target_all)], [min(target_all),
    max(target_all)], color="k", ls="--")
plt.xlim([min(target_all), max(target_all)])
plt.xlabel("True Values")
plt.ylabel("Predicted Values")

plt.show()

#%% Data Histogram

plt.figure(figsize=(4, 3), dpi=100)
plt.hist(data_set.df["-logKd/Ki"])
plt.xlabel("-logKd/Ki")
plt.ylabel("N")
plt.show()
