import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import os
import IPython.display as ipd
from tqdm import tqdm_notebook
import seaborn as sn
import torch
import torch.nn as nn
import random as rn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch.optim as optim
import glob
import sys 
from torch.autograd import Variable
import os
import pandas as pd
import glob
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
import os
import IPython.display as ipd
from tqdm import tqdm_notebook
import IPython.display as ipd

SEED = 2222 
np.random.seed(SEED)
rn.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)


try:
    data_folder = str(sys.argv[1])
except:
    print("Please enter test directory")
    data_folder = input()
glob_it = data_folder + '/*.wav'
files = glob.glob(glob_it)

file_names = []

for i in files:
    file_names.append(i.replace(str(data_folder) + '/',""))

sample_rate = 44100
dim = (100,1 + int(np.floor((sample_rate * 3)/512)))
input_length = sample_rate * 3
X_test = np.empty(shape=(len(files), dim[0], dim[1]))
X_test_1d = np.empty(shape=(len(files), dim[1]))

for i in range(len(files)):


    file_path = files[i]
    data, _ = librosa.load(file_path, res_type='kaiser_fast',duration=3,sr=22050*2,offset=0.5)
        
        # Random offset / Padding
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length+offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
                
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant") #padding 
            
    S = librosa.feature.melspectrogram(data,sr = 44100)
    log_S = librosa.power_to_db(S,ref = np.max)
    mfcc = librosa.feature.mfcc(S = log_S, sr=sample_rate, n_mfcc= dim[0])
    X_test[i,] = mfcc
    X_test_1d[i,] = np.mean(mfcc)

BATCH_SIZE = 300

X_test_tensor = torch.tensor(X_test)
X_test_tensor_1d = torch.tensor(X_test_1d)



class Convblock(nn.Module):  #for MFCC values
    def __init__(self,in_dim,out_dim,kernel,stride = 1,pool = (2,2)):
        super().__init__()
        self.conv = nn.Conv2d(in_dim,out_dim,kernel, stride=stride, padding=0)
        self.batchnorm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pool)
        self.avgpool = nn.AvgPool2d(pool)
       
    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x = self.avgpool(x)
        return x

class Convblock1D(nn.Module): #for mean values of MFCCs
    def __init__(self,in_dim,out_dim,kernel,stride = 1,pool = 2):
        super().__init__()
        self.conv = nn.Conv1d(in_dim,out_dim,kernel, stride=stride, padding=0)
        self.batchnorm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(pool)
        self.avgpool = nn.AvgPool1d(pool)
       
    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x = self.avgpool(x)
        return x



class CNN2D_CNN1D_GRU(nn.Module):
    def __init__(self,num_labels,convblock = Convblock,convblock1d = Convblock1D):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1,16,(3,5)), nn.ReLU(),nn.Dropout(0.2),nn.AvgPool2d(10,10))
        self.pool = nn.MaxPool2d(5,5)
        self.convolution1d = nn.Sequential(convblock1d(1,16,5),nn.Dropout(0.2),convblock1d(16,16,5),nn.Dropout(0.2))
        self.dense = nn.Sequential(nn.Linear( 4720,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Linear(64,num_labels))
        self.sigmoid = nn.Sigmoid()
        self.gru1 = nn.GRU(20,10,batch_first = True)
        self.gru2 = nn.GRU(16,10,batch_first = True)
        self.fc1 = nn.Linear(61*64,128)
        self.fc2 = nn.Linear(61*64,128)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x,y):
        x = self.dropout(x)
        z = self.conv(x)
        x = self.pool(x)
        x = x.squeeze(1)
        x = x.permute(0,2,1)
        x , hn = self.gru1(x)
        x = x.reshape(x.shape[0],-1)
        z = z.reshape(z.shape[0],-1)
        

        y = self.convolution1d(y)
        y = y.permute(0,2,1)
        y ,hn_y = self.gru2(y)
        y = y.reshape(y.shape[0],-1)
        a = torch.cat((x,z,y),dim = 1)
        a = self.dense(a)
        return a

model_for_eval = CNN2D_CNN1D_GRU(num_labels = 5).to(device)



submission = open('output.txt','w')

def print_values(pred_index,submission):
    if(pred_index == 0):
        submission.write(',happy \n')
    elif(pred_index == 1):
        submission.write(',sad \n')
    elif(pred_index == 2):
        submission.write(',disgust \n')
    elif(pred_index == 3):
        submission.write(',fear \n')
    elif(pred_index == 4):
        submission.write(',neutral \n')


try:
    MODEL_DIR = sys.argv[2]
except:
    print("Please provide the directory in which the model weights are saved")
    MODEL_DIR = input()
MODEL_DIR = os.path.join(MODEL_DIR, '2DCNN_1DCNN_GRU.pth')


if(torch.cuda.is_available() == False): 
    model_for_eval.load_state_dict(torch.load(MODEL_DIR,map_location=torch.device('cpu') ) )
else:
    model_for_eval.load_state_dict(torch.load(MODEL_DIR) )


model_for_eval.eval()
with torch.no_grad():
    for i,(mfcc,means) in enumerate(zip(X_test_tensor,X_test_tensor_1d)):
        mfcc = mfcc.unsqueeze(0).float().to(device)
        mfcc = mfcc.unsqueeze(1).float().to(device)
        means = means.unsqueeze(0).float().to(device)
        means = means.unsqueeze(1).float().to(device)
        logits = model_for_eval(mfcc,means)
        _ ,pred_index = logits[0].max(0)
        submission.write(str(file_names[i]))
        print_values(pred_index,submission)

        


submission.close()
print("DONE")





