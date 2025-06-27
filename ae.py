# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

import math



# Network definition
## Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.layers = {
            'block-1': {
                'avg-head': {},
                'lp-head': {},
                'max-head': {},
            },
            'block-2': {
                'avg-head': {},
                'lp-head': {},
                'max-head': {},
            },
            'block-3': {},
            'activation': nn.ReLU()
        }



        # Block 1
        ## AvgPool Head
        self.layers['block-1']['avg-head']['pool'] = nn.AvgPool1d(3, stride = 2, padding = 0)
        self.layers['block-1']['avg-head']['conv-1'] = nn.Conv1d(in_channels, 32, 3, stride = 1)
        self.layers['block-1']['avg-head']['conv-2'] = nn.Conv1d(32, 64, 3, stride = 1)

        ## LPPool Head
        self.layers['block-1']['lp-head']['pool'] = nn.AvgPool1d(3, stride = 2, padding = 0)
        self.layers['block-1']['lp-head']['conv-1'] = nn.Conv1d(in_channels, 32, 3, stride = 1)
        self.layers['block-1']['lp-head']['conv-2'] = nn.Conv1d(32, 64, 3, stride = 1)

        ## MaxPool Head
        self.layers['block-1']['max-head']['pool'] = nn.MaxPool1d(3, stride = 2, padding = 0)
        self.layers['block-1']['max-head']['conv-1'] = nn.Conv1d(in_channels, 32, 3, stride = 1)
        self.layers['block-1']['max-head']['conv-2'] = nn.Conv1d(32, 64, 3, stride = 1)

        ## Aggregator
        self.layers['block-1']['aggregator'] = nn.Conv1d(64 * 3, 128, 3, stride = 1)



        # Block 2
        ## AvgPool Head
        self.layers['block-2']['avg-head']['pool'] = nn.AvgPool1d(3, stride = 2, padding = 0)
        self.layers['block-2']['avg-head']['conv-1'] = nn.Conv1d(128, 32, 3, stride = 1)
        self.layers['block-2']['avg-head']['conv-2'] = nn.Conv1d(32, 64, 3, stride = 1)

        ## LPPool Head
        self.layers['block-2']['lp-head']['pool'] = nn.AvgPool1d(3, stride = 2, padding = 0)
        self.layers['block-2']['lp-head']['conv-1'] = nn.Conv1d(128, 32, 3, stride = 1)
        self.layers['block-2']['lp-head']['conv-2'] = nn.Conv1d(32, 64, 3, stride = 1)

        ## MaxPool Head
        self.layers['block-2']['max-head']['pool'] = nn.MaxPool1d(3, stride = 2, padding = 0)
        self.layers['block-2']['max-head']['conv-1'] = nn.Conv1d(128, 32, 3, stride = 1)
        self.layers['block-2']['max-head']['conv-2'] = nn.Conv1d(32, 64, 3, stride = 1)

        ## Aggregator
        self.layers['block-2']['aggregator'] = nn.Conv1d(64 * 3, 128, 3, stride = 1)



        # Block 3
        ## Embedder
        self.layers['block-3']['embedder'] = nn.Conv1d(128, out_channels, 3, stride = 1)


    # Aux function to apply pad 'sane' to strided convolutions
    def __pad(self, x, kernel, stride):
        L_in = x.size(-1)

        # Compute the total padding needed to ensure "same" output length
        L_out = math.ceil(L_in / stride)
        pad_needed = max(0, (L_out - 1) * stride + kernel - L_in)

        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left

        return F.pad(x, (pad_left, pad_right))

    def forward(self, x):
        # Block 1
        ## Avg Head
        avg_head_b1 = self.layers['block-1']['avg-head']['pool'](x)

        avg_head_b1 = self.__pad(avg_head_b1, self.layers['block-1']['avg-head']['conv-1'].kernel_size, self.layers['block-1']['avg-head']['conv-1'].strides)
        avg_head_b1 = self.layers['block-1']['avg-head']['conv-1'](avg_head_b1)
        avg_head_b1 = self.layers['activation'](avg_head_b1)

        avg_head_b1 = self.__pad(avg_head_b1, self.layers['block-1']['avg-head']['conv-2'].kernel_size, self.layers['block-1']['avg-head']['conv-2'].strides)
        avg_head_b1 = self.layers['block-1']['avg-head']['conv-2'](avg_head_b1)
        avg_head_b1 = self.layers['activation'](avg_head_b1)

        ## LP Head
        lp_head_b1 = self.layers['block-1']['lp-head']['pool'](x)

        lp_head_b1 = self.__pad(lp_head_b1, self.layers['block-1']['lp-head']['conv-1'].kernel_size, self.layers['block-1']['lp-head']['conv-1'].strides)
        lp_head_b1 = self.layers['block-1']['lp-head']['conv-1'](lp_head_b1)
        lp_head_b1 = self.layers['activation'](lp_head_b1)

        lp_head_b1 = self.__pad(lp_head_b1, self.layers['block-1']['lp-head']['conv-2'].kernel_size, self.layers['block-1']['lp-head']['conv-2'].strides)
        lp_head_b1 = self.layers['block-1']['lp-head']['conv-2'](lp_head_b1)
        lp_head_b1 = self.layers['activation'](lp_head_b1)

        ## Max Head
        max_head_b1 = self.layers['block-1']['max-head']['pool'](x)
        
        max_head_b1 = self.__pad(max_head_b1, self.layers['block-1']['max-head']['conv-1'].kernel_size, self.layers['block-1']['max-head']['conv-1'].strides)
        max_head_b1 = self.layers['block-1']['max-head']['conv-1'](max_head_b1)
        max_head_b1 = self.layers['activation'](max_head_b1)
        
        max_head_b1 = self.__pad(max_head_b1, self.layers['block-1']['max-head']['conv-2'].kernel_size, self.layers['block-1']['max-head']['conv-2'].strides)
        max_head_b1 = self.layers['block-1']['max-head']['conv-2'](max_head_b1)
        max_head_b1 = self.layers['activation'](max_head_b1)

        ## Aggregator
        concat_b1 = torch.cat([avg_head_b1, lp_head_b1, max_head_b1], dim=1)
        concat_b1 = self.__pad(concat_b1, self.layers['block-1']['aggregator'].kernel_size, self.layers['block-1']['aggregator'].strides)
        aggregated_b1 = self.layers['block-1']['aggregator'](concat_b1)
        aggregated_b1 = self.layers['activation'](aggregated_b1)
        


        # Block 2
        ## Avg Head
        avg_head_b2 = self.layers['block-2']['avg-head']['pool'](aggregated_b1)

        avg_head_b2 = self.__pad(avg_head_b2, self.layers['block-2']['avg-head']['conv-1'].kernel_size, self.layers['block-2']['avg-head']['conv-1'].strides)
        avg_head_b2 = self.layers['block-2']['avg-head']['conv-1'](avg_head_b2)
        avg_head_b2 = self.layers['activation'](avg_head_b2)

        avg_head_b2 = self.__pad(avg_head_b2, self.layers['block-2']['avg-head']['conv-2'].kernel_size, self.layers['block-2']['avg-head']['conv-2'].strides)
        avg_head_b2 = self.layers['block-2']['avg-head']['conv-2'](avg_head_b2)
        avg_head_b2 = self.layers['activation'](avg_head_b2)

        ## LP Head
        lp_head_b2 = self.layers['block-2']['lp-head']['pool'](aggregated_b1)

        lp_head_b2 = self.__pad(lp_head_b2, self.layers['block-2']['lp-head']['conv-1'].kernel_size, self.layers['block-2']['lp-head']['conv-1'].strides)
        lp_head_b2 = self.layers['block-2']['lp-head']['conv-1'](lp_head_b2)
        lp_head_b2 = self.layers['activation'](lp_head_b2)

        lp_head_b2 = self.__pad(lp_head_b2, self.layers['block-2']['lp-head']['conv-2'].kernel_size, self.layers['block-2']['lp-head']['conv-2'].strides)
        lp_head_b2 = self.layers['block-2']['lp-head']['conv-2'](lp_head_b2)
        lp_head_b2 = self.layers['activation'](lp_head_b2)

        ## Max Head
        max_head_b2 = self.layers['block-2']['max-head']['pool'](aggregated_b1)
        
        max_head_b2 = self.__pad(max_head_b2, self.layers['block-2']['max-head']['conv-1'].kernel_size, self.layers['block-2']['max-head']['conv-1'].strides)
        max_head_b2 = self.layers['block-2']['max-head']['conv-1'](max_head_b2)
        max_head_b2 = self.layers['activation'](max_head_b2)
        
        max_head_b2 = self.__pad(max_head_b2, self.layers['block-2']['max-head']['conv-2'].kernel_size, self.layers['block-2']['max-head']['conv-2'].strides)
        max_head_b2 = self.layers['block-2']['max-head']['conv-2'](max_head_b2)
        max_head_b2 = self.layers['activation'](max_head_b2)

        ## Aggregator
        concat_b2 = torch.cat([avg_head_b2, lp_head_b2, max_head_b2], dim=1)
        concat_b2 = self.__pad(concat_b2, self.layers['block-2']['aggregator'].kernel_size, self.layers['block-2']['aggregator'].strides)
        aggregated_b2 = self.layers['block-2']['aggregator'](concat_b2)
        aggregated_b2 = self.layers['activation'](aggregated_b2)
        


        # Block 3
        embedding = self.layers['block-3']['embedder'](aggregated_b2)

        return embedding



## Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.layers = {
            'block-1': {
                'up-tail': {},
                'deconv-tail': {}
            },
            'block-2': {
                'up-tail': {},
                'deconv-tail': {}
            },
            'skip-tail': {},
            'activation': nn.ReLU()
        }

        # Expander
        self.layers['expander'] = nn.Conv1d(in_channels, 256, 3, stride = 1)
        
        # Block 1
        self.layers['block-1']['up-tail'] = nn.Upsample(scale_factor = 2, mode = 'linear')

        self.layers['block-1']['deconv-tail']['conv-t'] = nn.ConvTranspose1d(256, 32, 3, stride = 2, padding = 0)
        self.layers['block-1']['deconv-tail']['conv'] = nn.Conv1d(in_channels, 256, 3, stride = 1)
        self.layers['block-1']['aggregator'] = nn.Conv1d(in_channels, 256, 3, stride = 1)
        
        # Block 2
        self.layers['block-2']['up-tail'] = nn.Upsample(scale_factor = 2, mode = 'linear')
        
        self.layers['block-2']['deconv-tail']['conv-t'] = nn.ConvTranspose1d(256, 32, 3, stride = 2, padding = 0)
        self.layers['block-2']['deconv-tail']['conv'] = nn.Conv1d(in_channels, 256, 3, stride = 1)
        self.layers['block-2']['aggregator'] = nn.Conv1d(in_channels, 256, 3, stride = 1)
        
        # Skip Tail
        self.layers['skip-tail']['conv-t-1'] = nn.ConvTranspose1d(256, 32, 3, stride = 2, padding = 0)
        self.layers['skip-tail']['conv-t-2'] = nn.ConvTranspose1d(256, 32, 3, stride = 2, padding = 0)

        # Aggregator
        self.layers['aggregator'] = nn.Conv1d(in_channels, out_channels, 3, stride = 1)


    # Aux function to apply pad 'sane' to strided convolutions
    def __pad(self, x, kernel, stride):
        L_in = x.size(-1)

        # Compute the total padding needed to ensure "same" output length
        L_out = math.ceil(L_in / stride)
        pad_needed = max(0, (L_out - 1) * stride + kernel - L_in)

        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left

        return F.pad(x, (pad_left, pad_right))

    def forward(self, x):
        # Expander
        x = self.__pad(x, self.layers['expander'].kernel, self.layers['expander'].stride)
        expanded = self.layers['expander'](x)
        expanded = self.layers['activation'](expanded)
        
        # Block 1
        up_tail_b1 = self.layers['block-1']['up-tail'](expanded)

        deconv_tail_b1 = self.layers['block-1']['deconv-tail']['conv-t'](expanded)
        deconv_tail_b1 = self.layers['activation'](deconv_tail_b1)
        deconv_tail_b1 = self.__pad(deconv_tail_b1, self.layers['block-1']['deconv-tail']['conv'].kernel, self.layers['block-1']['deconv-tail']['conv'].stride)
        deconv_tail_b1 = self.layers['block-1']['deconv-tail']['conv'](deconv_tail_b1)
        deconv_tail_b1 = self.layers['activation'](deconv_tail_b1)

        add_b1 = up_tail_b1 + deconv_tail_b1
        add_b1 = self.__pad(add_b1, self.layers['block-1']['aggregator'].kernel, self.layers['block-1']['aggregator'].stride)
        aggregator_b1 = self.layers['block-1']['aggregator'](add_b1)
        aggregator_b1 = self.layers['activation'](aggregator_b1)
        

        
        # Block 2
        up_tail_b2 = self.layers['block-1']['up-tail'](expanded)

        deconv_tail_b2 = self.layers['block-2']['deconv-tail']['conv-t'](aggregator_b1)
        deconv_tail_b2 = self.layers['activation'](deconv_tail_b2)
        deconv_tail_b2 = self.__pad(deconv_tail_b2, self.layers['block-2']['deconv-tail']['conv'].kernel, self.layers['block-2']['deconv-tail']['conv'].stride)
        deconv_tail_b2 = self.layers['block-2']['deconv-tail']['conv'](deconv_tail_b2)
        deconv_tail_b2 = self.layers['activation'](deconv_tail_b2)

        add_b2 = up_tail_b2 + deconv_tail_b2
        aggregator_b2 = self.layers['block-2']['aggregator'](add_b2)
        
        # Skip Tail
        skip_tail = self.layers['skip-tail']['conv-t-1'](expanded)
        skip_tail = self.layers['activation'](skip_tail)
        skip_tail = self.layers['skip-tail']['conv-t-2'](skip_tail)
        skip_tail = self.layers['activation'](skip_tail)

        # Aggregator
        concat = torch.cat([aggregator_b2, skip_tail], dim=1)
        concat = self.__pad(concat, self.layers['aggregator'].kernel, self.layers['aggregator'].stride)
        out = self.layers['aggregator'](concat)
        
        return out



## AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self, in_channels):
        super(AutoEncoder, self).__init__()
        
        # Components
        self.encoder = Encoder(in_channels, 64)
        self.decoder = Decoder(64, in_channels)

    def forward(self, x):
        # Encode
        embedding = self.encoder(x)
        
        # Decode
        out = self.decoder(embedding)

        return out
