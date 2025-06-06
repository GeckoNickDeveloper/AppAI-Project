# Imports
import torch.nn as nn

# Network definition
## Encoder
class Encoder(nn.Module):
    """ TODO: add doc """
    def __init__(self):
        super(Encoder, self).__init__()

        # Block 1
        self.max_pool_b1 = nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False) # TODO: Tune params
        self.avg_pool_b1 = nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) # TODO: Tune params
        self.conv_b1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) # TODO: Tune params
        
        # Block 2
        self.max_pool_b2 = nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False) # TODO: Tune params
        self.avg_pool_b2 = nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) # TODO: Tune params
        self.conv_b2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) # TODO: Tune params

        # Block 3
        self.conv_b3 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) # TODO: Tune params

        self.lp_pool_b3 = nn.LPPool1d(norm_type, kernel_size, stride=None, ceil_mode=False) # TODO: Tune params 

    """ TODO: add doc """
    def forward(self, x):
        pass



## Decoder
class Decoder(nn.Module):
    """ TODO: add doc """
    def __init__(self):
        super(Decoder, self).__init__()

        # Block 1
        self.up_b1 = 'todo'
        self.convt1_b1 = 'todo'
        self.convt2_b1 = 'todo'

        # Block 2
        self.up_b2 = 'todo'
        self.convt1_b2 = 'todo'
        self.convt2_b2 = 'todo' 

        # Block 3
        self.embedding_up = 'todo'
        self.conv1_b3 = 'todo'
        self.conv2_b3 = 'todo'

    """ TODO: add doc """
    def forward(self, x):
        pass



## AutoEncoder
class AutoEncoder(nn.Module):
    """ TODO: add doc """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # Components
        self.encoder = Encoder()
        self.decoder = Decoder()

    """ TODO: add doc """
    def forward(self, x):
        # Encode
        embedding = self.encoder(x)
        
        # Decode
        out = self.decoder(embedding)

        return out
