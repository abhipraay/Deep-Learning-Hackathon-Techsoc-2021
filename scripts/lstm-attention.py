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

        
      
        self.title_lstm = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, bidirectional = True, batch_first = True)
        self.content_lstm = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, bidirectional = True, batch_first = True)
        
        self.attention = nn.Linear(2*self.hidden_size, 2*self.hidden_size, bias = False)
        
        self.fc = nn.Linear(637, self.num_labels)
        self.fc1 = nn.Linear(4*hidden_size, self.num_labels)
        
    def forward(self, content, title):
        
        content_embed = self.embed(content)
        title_embed = self.embed(title)
       
        title_rep, _ = self.title_lstm(title_embed)                      #[64, 31, 400]
        content_rep, _ = self.content_lstm(content_embed)                      #[64,637, 400]
        
        
        
        
        title_rep = title_rep[:,-1,:].unsqueeze(1)                       #[64,1,400]
        a = []
        for i in range(content_rep.shape[1]):
            c = self.attention(content_rep[:,i,:]).unsqueeze(2)                # [64, 400, 1]
            c2 = []
            for batch in range(content_rep.shape[0]):
                c1 = title_rep[batch,:,:] @ c[batch,:,:]  # [1,1]
                c2.append(c1)
            c2 = torch.stack(c2)                          #[64,1,1]
            a.append(c2)                              
        
        a = torch.stack(a)           
        a = a.permute(1,0,2,3).squeeze(-1).squeeze(-1)

        a = F.softmax(a, dim = 1)    # [64,500]
        preds = F.log_softmax(self.fc(a), dim = 1)    #[64,500]
        
        
        return preds
 