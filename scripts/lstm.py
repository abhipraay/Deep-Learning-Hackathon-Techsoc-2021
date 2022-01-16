import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    
    def __init__(self, embed_size, hidden_size,  vocab_size, num_labels, weights_matrix ):
        super().__init__()
        
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        self.embed = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.embed_size)
        self.embed.weight.requires_grad = False
        self.embed.load_state_dict({'weight': weights_matrix})

        
      
        self.title_lstm = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, bidirectional = True, batch_first = True, num_layers = 2)
        self.content_lstm = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, bidirectional = True, batch_first = True, num_layers = 2)
        
               
        self.fc = nn.Linear(4*hidden_size, self.num_labels)
        self.relu = nn.ReLU()
        
    def forward(self, content, title):
        
        content_embed = self.embed(content)
        title_embed = self.embed(title)
       
        t_rep, _ = (self.title_lstm(title_embed))                      
        c_rep, _ = (self.content_lstm(content_embed))                   
        title_rep = self.relu(t_rep)
        content_rep = self.relu(c_rep)
        

        t = torch.mean(title_rep, dim = 1)
        c = torch.mean(content_rep, dim = 1)
        final_rep = torch.cat((t,c), dim = 1)
        return F.log_softmax((self.fc(final_rep)), dim = 1)