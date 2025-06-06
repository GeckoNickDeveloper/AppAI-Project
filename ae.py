# Imports
import torch.nn as nn

# Network definition
## Encoder
class Encoder(nn.Module):
    """ TODO: add doc """
    def __init__(self, arg):
        super(Encoder, self).__init__()
        self.arg = arg

        # Block 1
        self.max_pool_b1 = 'todo'
        self.avg_pool_b1 = 'todo'
        self.conv_b1 = 'todo'
        
        # Block 2
        self.max_pool_b2 = 'todo'
        self.avg_pool_b2 = 'todo'
        self.conv_b2 = 'todo'

        # Block 3
        self.conv_b3 = 'todo'
        self.lp_pool_b3 = 'todo'

    """ TODO: add doc """
    def forward(self, inputs):
        pass



## Decoder
class Decoder(nn.Module):
    """ TODO: add doc """
    def __init__(self, arg):
        super(Decoder, self).__init__()
        self.arg = arg

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
    def forward(self, inputs):
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
    def forward(self, inputs):
        # Encode
        embedding = self.encoder(inputs)
        # Decode
        out = self.decoder(embedding)

        return out
