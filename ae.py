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
        # Block 1
        mp_b1 = self.max_pool_b1(x)
        ap_b1 = self.avg_pool_b1(x)
        conv_b1 = self.conv_b1(x)

        concat_b1 = torch.cat([mp_b1, conv_b1, ap_b1], dim=1)

        # Block 2
        mp_b2 = self.max_pool_b1(concat_b1)
        ap_b2 = self.avg_pool_b1(concat_b1)
        conv_b2 = self.conv_b1(concat_b1)

        concat_b2 = torch.cat([mp_b2, conv_b2, ap_b2], dim=1)

        # Block 3
        conv_b3 = self.conv_b3(concat_b2)
        lppool_b3 = self.lp_pool_b3(conv_b3)

        return lp_pool_b3




## Decoder
class Decoder(nn.Module):
    """ TODO: add doc """
    def __init__(self):
        super(Decoder, self).__init__()

        # Block 1
        self.up_b1 = nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
        self.convt1_b1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None) # TODO: Tune params
        self.convt2_b1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None) # TODO: Tune params

        # Block 2
        self.up_b2 = nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None) # TODO: Tune params
        self.convt1_b2 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None) # TODO: Tune params
        self.convt2_b2 =  nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None) # TODO: Tune params

        # Block 3
        self.embedding_up = nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None) # TODO: Tune params
        self.conv1_b3 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) # TODO: Tune params

        self.conv2_b3 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) # TODO: Tune params

    """ TODO: add doc """
    def forward(self, x):
        # Block 1
        up_b1 = self.up_b1(x)
        ct1_b1 = self.convt1_b1(x)
        ct2_b1 = self.convt2_b1(ct1_b1)

        concat_b1 = torch.cat([up_b1, ct2_b1], dim=1)

        # Block 2
        up_b2 = self.up_b1(x)
        ct1_b2 = self.convt1_b2(x)
        ct2_b2 = self.convt2_b2(ct1_b2)

        concat_b2 = torch.cat([up_b2, ct2_b2], dim=1) 

        # Block 3
        ## Embedding long dependency
        embed_up = self.embedding_up(x)
        
        ## Decoder terminal
        c1_b3 = self.conv1_b3(concat_b2)

        ## Combine upscaled embeding and decoders
        add_b3 = c1_b3 + embed_up

        ## Transform data 
        c2_b3 = self.conv2_b3(add_b3)

        return c2_b3




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
