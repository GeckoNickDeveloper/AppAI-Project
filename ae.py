# Imports
import torch.nn as nn

# Network definition
## Encoder
class Encoder(nn.Module):
    """ TODO: add doc """
    def __init__(self, arg):
        super(Encoder, self).__init__()
        self.arg = arg

    """ TODO: add doc """
    def forward(self, inputs):
        pass



## Decoder
class Decoder(nn.Module):
    """ TODO: add doc """
    def __init__(self, arg):
        super(Decoder, self).__init__()
        self.arg = arg

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
