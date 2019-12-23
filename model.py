import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        # showing error w/o this: AttributeError: cannot assign module before Module.__init__() call
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, features, captions, cuda=True):
        # <end> isn't required in the input
        caption_embedding = self.embed(captions[:, :-1])

        '''
        features - 10x256 - batch x embed_size
        caption_embedding - 10x15x256 - batch x caption_length x embed_size
        
        Resize features to - 10x1x256
        '''
        features = features.view(features.shape[0], 1, features.shape[1])
        
        inputs = torch.cat([features, caption_embedding], dim=1)
        
        # hidden - (num_layers, batch_size, hidden_size)
        if cuda == True:
            hidden = (torch.randn(self.num_layers, inputs.shape[0], self.hidden_size).cuda(),
                      torch.randn(self.num_layers, inputs.shape[0], self.hidden_size).cuda())
        else:
            hidden = (torch.randn(self.num_layers, inputs.shape[0], self.hidden_size),
                      torch.randn(self.num_layers, inputs.shape[0], self.hidden_size))
            
        out, hidden = self.lstm(inputs, hidden)
        
        out = self.linear(out)
        
        return out
        

    def sample(self, inputs, states=None, max_len=20, cuda=True):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        if cuda == True:
            hidden = (torch.randn(self.num_layers, inputs.shape[0], self.hidden_size).cuda(),
                      torch.randn(self.num_layers, inputs.shape[0], self.hidden_size).cuda())
        else:
            hidden = (torch.randn(self.num_layers, inputs.shape[0], self.hidden_size),
                      torch.randn(self.num_layers, inputs.shape[0], self.hidden_size))
            
        outputs = []
        
        count = 0
        while count <= max_len:# and (out_token.item() != 1)
            if count == 0:
                embedding = inputs
            else:
                embeddding = self.embed(out_token) # embedding: 1x1x256


        #     print(embedding.shape, hidden[0].shape)
            out, hidden = self.lstm(embedding, hidden)

            token_scores =  self.linear(out) # token_scores : 1xvocab_size
            out_token = token_scores.topk(1)[1].squeeze(0) # token1 : 1x1

            outputs.append(out_token.item())

            count += 1
        
        reshape to 1x1xembed_size
        inputs = inputs.unsqueeze(0).unsqueeze(0)

        
        return outputs
        
        