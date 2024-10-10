import math
import torch 
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model : int , vocab_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size , d_model) #d_model is the embedding vector size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PosititonalEncoding(nn.Module):
    def __init__(self, d_model: int , seq_len: int , dropout : float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) #Used to add some bias to the vector by randomly zeroing some elements in each foward pass

        # Creating a matrix of shape (seq_len , d_model)
        pe = torch.zeros(seq_len , d_model)
        # Creating a vector of shape (seq_len , 1)
        position = torch.arange(0 , seq_len , dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0 , d_model , 2).float() * (-math.log(10000,0)/d_model))
        #Apply sin to even pos
        pe[: , 0 ::2] = torch.sin(position * div_term)
        #Apply cos to odd pos
        pe[: , 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # adds a dimension at index 0 (1, Seq_len , d_model)

        self.register_buffer('pe' , pe) # saved in the file, not as a learnable parameter

    def forward(self, x):
        x = x + (self.pe[: , :x.shape[1] , :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalisation(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #Added

    def forward(self , x):
        mean = x.mean(dim = -1 , keepdim = True)
        std = x.std(dim = -1 , keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model : int , d_ff : int , dropout : float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model , d_ff) # both weights and biases
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff , d_model) # both weights and biases

    def forward(self , x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
vocab_size = 50000
d_model = 512
model = InputEmbeddings(d_model , vocab_size)
input = ["Your" , "cat" , "is" , "a" , "Lovely" , "cat"]
input = torch.tensor(input)
output = model(input)
print(output)