import torch
import torch.nn as nn

class Mynetwork(nn.Module):
    def __init__(self,input_num=3 , out_num=1,hidden_num=32):
        super().__init__()
        self.MLP=nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(input_num,hidden_num),
            nn.LeakyReLU(),
            nn.Linear(hidden_num,out_num)
        )
        pass
    def forward(self,f,t,dt):
        input_tensor=torch.cat((f,t,dt),dim=-1)
        return self.MLP(input_tensor)

class kernelNN(nn.Module):
    def __init__(self, fn_dim=1, var_dim=1, output_dim=1, hidden_dim=32):
        super(kernelNN, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(fn_dim+2*var_dim, hidden_dim),  # Concatenate particle and row position embeddings
                                 nn.LeakyReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),  # Concatenate particle and row position embeddings
                                 nn.LeakyReLU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, fn, x, dx):
        input = torch.cat((fn, x, dx), dim=-1)
        return self.mlp(input)