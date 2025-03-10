import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.transformer import MultiheadAttention, _get_activation_fn
from torch.nn import functional as f
import torch.nn as nn





class classifier_generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(classifier_generator, self).__init__()
        self.shared_net = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU()
                        )
        self.shared_mu = nn.Linear(hidden_size, output_size)
        self.shared_sigma = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        z = self.shared_net(x)
        return self.shared_mu(z), f.softplus(self.shared_sigma(z), beta=1, threshold=20)

class TransformerModel_67(nn.Module):
    def __init__(self, feature_dim, num_class, nhead=2, nhid=512, nlayers=8, batch_first=True, dropout=0.1, adp_sam='s', domemb=False):
        super(TransformerModel_67, self).__init__()
        self.feature_dim = 512
        self.num_class = num_class
        self.domemb = domemb
        self.adp_sam = adp_sam
        if self.adp_sam == 'm':
            batch_first = False
            
        encoder_layers = TransformerEncoderLayer(feature_dim * 2**(int(self.domemb)), nhead, nhid, batch_first=batch_first)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.weights_net = classifier_generator(feature_dim, feature_dim, feature_dim)
        self.feature_net = nn.Sequential(
                        nn.Linear(feature_dim, feature_dim),
                        nn.ReLU(), 
                        nn.Linear(feature_dim, feature_dim)
                        )
    def forward(self, ctx, features, ensemble=False):
        
        if ctx is None:
            return self.feature_net(features), self.feature_net(features)
        if 1==2:
            features = features.view(features.size()[0], 1, features.size()[1])    
            ctx = ctx.view(1, self.num_class, self.feature_dim).repeat(features.size()[0], 1, 1) 
            outputs = self.transformer_encoder(torch.cat([ctx, features], 1))
            
            weights_mu, weights_sigma = self.weights_net(outputs[:, :-1, :].contiguous().view(-1, self.feature_dim))
            weights_mu = weights_mu.view(features.size()[0], self.num_class, self.feature_dim)
            weights_sigma = f.softplus(weights_sigma, beta=1, threshold=20).view(features.size()[0], ctx.size()[1], self.feature_dim)
            feats = self.feature_net(outputs[:, -1, :])

            return weights_mu, weights_sigma

        else:  
            
            
            
            

            mask = self.generate_D_q_matrix(ctx.size()[0] + features.size()[0], features.size()[0]).to(features.device)
            
            
            
            
            
            outputs = self.transformer_encoder(torch.cat([ctx, features], 0), mask)
            
            
            
            
            weights_mu, weights_sigma = self.weights_net(outputs[:self.num_class, :].contiguous().view(-1, self.feature_dim))  
            weights_mu = weights_mu.view(self.num_class, self.feature_dim)
            weights_sigma = f.softplus(weights_sigma, beta=1, threshold=20).view(self.num_class, self.feature_dim)
            feats = self.feature_net(outputs[self.num_class:, :])

            
            
            
            
            

            return weights_mu, weights_sigma

        

        
    def generate_D_q_matrix(self, sz, query_size):
        train_size = sz-query_size
        mask = torch.zeros(sz,sz) == 0
        mask[:,train_size:].zero_()
        mask |= torch.eye(sz) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask