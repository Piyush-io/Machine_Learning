import math
import torch 
import torch.nn as nn

class InputEmbeddings(nn.Module) :

    def __init__(self, d_model : int , vocab_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size , d_model) #d_model is the embedding vector size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PosititonalEncoding(nn.Module):
    def __init__(self, d_model: int , seq_len: int , dropout : float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) #Used to add some bias to the vector by randomly zeroing some elements in each foward pass

        # Creating a matrix of shape (seq_len , d_model)
        pe = torch.zeros(seq_len , d_model)
        # Creating a vector of shape (seq_len , 1)
        position = torch.arange(0 , seq_len , dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0 , d_model , 2).float() * (-math.log(10000,0)/d_model))

    

vocab_size = 50000
d_model = 512
model = InputEmbeddings(d_model , vocab_size)
input = ["Your" , "cat" , "is" , "a" , "Lovely" , "cat"]
input = torch.tensor(input)
output = model(input)
print(output)