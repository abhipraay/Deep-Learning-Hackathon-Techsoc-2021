import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_CNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size,  vocab_size, num_labels, weights_matrix, seq_len ):
        super().__init__()
        
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.seq_len = seq_len
        
        self.embed = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.embed_size)
        self.embed.weight.requires_grad = False
        self.embed.load_state_dict({'weight': weights_matrix})

        
      
    
        self.lstm = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, bidirectional = True, batch_first = True, num_layers = 2)
        
        self.conv = nn.Conv1d(2*self.hidden_size,64, 3)
        self.avg_pool = nn.AvgPool1d(seq_len - 2)
        self.max_pool = nn.MaxPool1d(seq_len - 2)
        self.fc = nn.Linear(128, self.num_labels)
        self.relu = nn.ReLU()
        
    def forward(self, info):
        
        info_embed = self.embed(info)
        i_rep, _ = (self.lstm(info_embed))                      #[32,seq_len, 128]
        info_rep = self.relu(i_rep)                             #[32,seq_len,128]
        info_conv = self.conv(info_rep.permute(0,2,1))          #[32,64,seq_len-2]  
        info_avg = self.avg_pool(info_conv).squeeze(-1)         #[32,64]
        info_max = self.max_pool(info_conv).squeeze(-1)         #[32,64]
        info_cat = torch.cat((info_avg, info_max), dim = 1)     #[32,128]
        

        return F.log_softmax(self.fc(info_cat), dim = 1)