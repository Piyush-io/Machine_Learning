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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model , h: int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisble by h"
        self.d_k= d_model // h
        self.w_q = nn.Linear(d_model , d_model) 
        self.w_k = nn.Linear(d_model , d_model)       
        self.w_v = nn.Linear(d_model , d_model)       
        self.w_o = nn.Linear(d_model , d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query , key , value , mask , dropout : nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0 , -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch , h , seq_len , seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)\
        
        return ( attention_scores @ value ) , attention_scores

        
    def forward(self , q , k , v , mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0] , query[1] , self.h , self.d_k).transpose(1,2) #we keep batch dimension , seq  and split the embedding to h and d_k
        key = key.view(key.shape[0] , key.shape[1] , self.h , self.d_k).transpose(1,2)
        value = value.view(value.shape[0] , value.shape[1] , self.h , self.d_k).transpose(1,2)

        x , self.attention_scores = MultiHeadAttention.attention(query , key , value , mask , self.dropout)


vocab_size = 50000
d_model = 512
model = InputEmbeddings(d_model , vocab_size)
input = ["Your" , "cat" , "is" , "a" , "Lovely" , "cat"]
input = torch.tensor(input)
output = model(input)
print(output)