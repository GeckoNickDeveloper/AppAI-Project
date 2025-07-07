# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import logger

# Network definition
## Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels: int, filters: int, out_channels: int):
        super(Encoder, self).__init__()

        # Logger
        self.logger = logger.get_logger(self.__class__.__name__, logger.WARNING)

        # Layers
        self.layers = nn.ModuleDict({
            'block-1': nn.ModuleDict({
                'avg-head': nn.ModuleDict({}),
                'lp-head': nn.ModuleDict({}),
                'max-head': nn.ModuleDict({}),
            }),
            'block-2': nn.ModuleDict({
                'avg-head': nn.ModuleDict({}),
                'lp-head': nn.ModuleDict({}),
                'max-head': nn.ModuleDict({}),
            }),
            'block-3': nn.ModuleDict({}),
            'activation': nn.ReLU()
        })



        # Block 1
        ## AvgPool Head
        self.layers['block-1']['avg-head']['pool'] = nn.AvgPool1d(2, stride = 2, padding = 0)
        self.layers['block-1']['avg-head']['conv-1'] = nn.Conv1d(in_channels, filters, 3, stride = 1)
        self.layers['block-1']['avg-head']['conv-2'] = nn.Conv1d(filters, filters * 2, 3, stride = 1)

        ## LPPool Head
        self.layers['block-1']['lp-head']['pool'] = nn.LPPool1d(5, 2, stride = 2)
        self.layers['block-1']['lp-head']['conv-1'] = nn.Conv1d(in_channels, filters, 3, stride = 1)
        self.layers['block-1']['lp-head']['conv-2'] = nn.Conv1d(filters, filters * 2, 3, stride = 1)

        ## MaxPool Head
        self.layers['block-1']['max-head']['pool'] = nn.MaxPool1d(2, stride = 2, padding = 0)
        self.layers['block-1']['max-head']['conv-1'] = nn.Conv1d(in_channels, filters, 3, stride = 1)
        self.layers['block-1']['max-head']['conv-2'] = nn.Conv1d(filters, filters * 2, 3, stride = 1)

        ## Aggregator
        self.layers['block-1']['aggregator'] = nn.Conv1d(filters * 2 * 3, filters * 8, 3, stride = 1)



        # Block 2
        ## AvgPool Head
        self.layers['block-2']['avg-head']['pool'] = nn.AvgPool1d(3, stride = 2, padding = 0)
        self.layers['block-2']['avg-head']['conv-1'] = nn.Conv1d(filters * 8, filters, 3, stride = 1)
        self.layers['block-2']['avg-head']['conv-2'] = nn.Conv1d(filters, filters * 2, 3, stride = 1)

        ## LPPool Head
        self.layers['block-2']['lp-head']['pool'] = nn.LPPool1d(5, 2, stride = 2)
        self.layers['block-2']['lp-head']['conv-1'] = nn.Conv1d(filters * 8, filters, 3, stride = 1)
        self.layers['block-2']['lp-head']['conv-2'] = nn.Conv1d(filters, filters * 2, 3, stride = 1)

        ## MaxPool Head
        self.layers['block-2']['max-head']['pool'] = nn.MaxPool1d(3, stride = 2, padding = 0)
        self.layers['block-2']['max-head']['conv-1'] = nn.Conv1d(filters * 8, filters, 3, stride = 1)
        self.layers['block-2']['max-head']['conv-2'] = nn.Conv1d(filters, filters * 2, 3, stride = 1)

        ## Aggregator
        self.layers['block-2']['aggregator'] = nn.Conv1d(filters * 2 * 3, filters * 8, 3, stride = 1)



        # Block 3
        ## Embedder
        self.layers['block-3']['embedder'] = nn.Conv1d(filters * 8, out_channels, 3, stride = 1)

    # Aux function to apply pad 'sane' to strided convolutions
    def __pad(self, x, kernel, stride):
        L_in = x.size(-1)

        # Compute the total padding needed to ensure "same" output length
        L_out = math.ceil(L_in / stride)
        pad_needed = max(0, (L_out - 1) * stride + kernel - L_in)

        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left

        return F.pad(x, (pad_left, pad_right), mode = 'reflect')

    def forward(self, x):
        # Block 1
        x = self.__pad(x, self.layers['block-1']['avg-head']['pool'].kernel_size[0], self.layers['block-1']['avg-head']['pool'].stride[0])

        ## Avg Head
        avg_head_b1 = self.layers['block-1']['avg-head']['pool'](x)

        avg_head_b1 = self.__pad(avg_head_b1, self.layers['block-1']['avg-head']['conv-1'].kernel_size[0], self.layers['block-1']['avg-head']['conv-1'].stride[0])
        avg_head_b1 = self.layers['block-1']['avg-head']['conv-1'](avg_head_b1)
        avg_head_b1 = self.layers['activation'](avg_head_b1)

        avg_head_b1 = self.__pad(avg_head_b1, self.layers['block-1']['avg-head']['conv-2'].kernel_size[0], self.layers['block-1']['avg-head']['conv-2'].stride[0])
        avg_head_b1 = self.layers['block-1']['avg-head']['conv-2'](avg_head_b1)
        avg_head_b1 = self.layers['activation'](avg_head_b1)

        self.logger.debug(f'Avg Head (1) - {avg_head_b1.shape}')

        ## LP Head
        cx = torch.clamp(x, min=1e-6)
        lp_head_b1 = self.layers['block-1']['lp-head']['pool'](cx)

        lp_head_b1 = self.__pad(lp_head_b1, self.layers['block-1']['lp-head']['conv-1'].kernel_size[0], self.layers['block-1']['lp-head']['conv-1'].stride[0])
        lp_head_b1 = self.layers['block-1']['lp-head']['conv-1'](lp_head_b1)
        lp_head_b1 = self.layers['activation'](lp_head_b1)

        lp_head_b1 = self.__pad(lp_head_b1, self.layers['block-1']['lp-head']['conv-2'].kernel_size[0], self.layers['block-1']['lp-head']['conv-2'].stride[0])
        lp_head_b1 = self.layers['block-1']['lp-head']['conv-2'](lp_head_b1)
        lp_head_b1 = self.layers['activation'](lp_head_b1)
        
        self.logger.debug(f'LP Head (1) - {lp_head_b1.shape}')

        ## Max Head
        max_head_b1 = self.layers['block-1']['max-head']['pool'](x)
        
        max_head_b1 = self.__pad(max_head_b1, self.layers['block-1']['max-head']['conv-1'].kernel_size[0], self.layers['block-1']['max-head']['conv-1'].stride[0])
        max_head_b1 = self.layers['block-1']['max-head']['conv-1'](max_head_b1)
        max_head_b1 = self.layers['activation'](max_head_b1)
        
        max_head_b1 = self.__pad(max_head_b1, self.layers['block-1']['max-head']['conv-2'].kernel_size[0], self.layers['block-1']['max-head']['conv-2'].stride[0])
        max_head_b1 = self.layers['block-1']['max-head']['conv-2'](max_head_b1)
        max_head_b1 = self.layers['activation'](max_head_b1)
        
        self.logger.debug(f'Max Head (1) - {max_head_b1.shape}')

        ## Aggregator
        concat_b1 = torch.cat([avg_head_b1, lp_head_b1, max_head_b1], dim=1)
        concat_b1 = self.__pad(concat_b1, self.layers['block-1']['aggregator'].kernel_size[0], self.layers['block-1']['aggregator'].stride[0])
        aggregated_b1 = self.layers['block-1']['aggregator'](concat_b1)
        aggregated_b1 = self.layers['activation'](aggregated_b1)
        
        self.logger.debug(f'Aggregator (1) - {aggregated_b1.shape}')


        # Block 2
        aggregated_b1 = self.__pad(aggregated_b1, self.layers['block-2']['avg-head']['pool'].kernel_size[0], self.layers['block-2']['avg-head']['pool'].stride[0])
        
        ## Avg Head
        avg_head_b2 = self.layers['block-2']['avg-head']['pool'](aggregated_b1)

        avg_head_b2 = self.__pad(avg_head_b2, self.layers['block-2']['avg-head']['conv-1'].kernel_size[0], self.layers['block-2']['avg-head']['conv-1'].stride[0])
        avg_head_b2 = self.layers['block-2']['avg-head']['conv-1'](avg_head_b2)
        avg_head_b2 = self.layers['activation'](avg_head_b2)

        avg_head_b2 = self.__pad(avg_head_b2, self.layers['block-2']['avg-head']['conv-2'].kernel_size[0], self.layers['block-2']['avg-head']['conv-2'].stride[0])
        avg_head_b2 = self.layers['block-2']['avg-head']['conv-2'](avg_head_b2)
        avg_head_b2 = self.layers['activation'](avg_head_b2)

        self.logger.debug(f'Avg Head (2) - {avg_head_b2.shape}')
        
        ## LP Head
        c_aggregated_b1 = torch.clamp(aggregated_b1, min=1e-6)
        lp_head_b2 = self.layers['block-2']['lp-head']['pool'](c_aggregated_b1)

        lp_head_b2 = self.__pad(lp_head_b2, self.layers['block-2']['lp-head']['conv-1'].kernel_size[0], self.layers['block-2']['lp-head']['conv-1'].stride[0])
        lp_head_b2 = self.layers['block-2']['lp-head']['conv-1'](lp_head_b2)
        lp_head_b2 = self.layers['activation'](lp_head_b2)

        lp_head_b2 = self.__pad(lp_head_b2, self.layers['block-2']['lp-head']['conv-2'].kernel_size[0], self.layers['block-2']['lp-head']['conv-2'].stride[0])
        lp_head_b2 = self.layers['block-2']['lp-head']['conv-2'](lp_head_b2)
        lp_head_b2 = self.layers['activation'](lp_head_b2)

        self.logger.debug(f'LP Head (2) - {lp_head_b2.shape}')
        
        ## Max Head
        max_head_b2 = self.layers['block-2']['max-head']['pool'](aggregated_b1)
        
        max_head_b2 = self.__pad(max_head_b2, self.layers['block-2']['max-head']['conv-1'].kernel_size[0], self.layers['block-2']['max-head']['conv-1'].stride[0])
        max_head_b2 = self.layers['block-2']['max-head']['conv-1'](max_head_b2)
        max_head_b2 = self.layers['activation'](max_head_b2)
        
        max_head_b2 = self.__pad(max_head_b2, self.layers['block-2']['max-head']['conv-2'].kernel_size[0], self.layers['block-2']['max-head']['conv-2'].stride[0])
        max_head_b2 = self.layers['block-2']['max-head']['conv-2'](max_head_b2)
        max_head_b2 = self.layers['activation'](max_head_b2)

        self.logger.debug(f'Max Head (2) - {max_head_b2.shape}')

        ## Aggregator
        concat_b2 = torch.cat([avg_head_b2, lp_head_b2, max_head_b2], dim=1)
        concat_b2 = self.__pad(concat_b2, self.layers['block-2']['aggregator'].kernel_size[0], self.layers['block-2']['aggregator'].stride[0])
        aggregated_b2 = self.layers['block-2']['aggregator'](concat_b2)
        aggregated_b2 = self.layers['activation'](aggregated_b2)
        
        self.logger.debug(f'Aggregator (2) - {aggregated_b2.shape}')



        # Block 3
        aggregated_b2 = self.__pad(aggregated_b2, self.layers['block-3']['embedder'].kernel_size[0], self.layers['block-3']['embedder'].stride[0])
        embedding = self.layers['block-3']['embedder'](aggregated_b2)

        self.logger.debug(f'Embedder - {embedding.shape}')

        return embedding



## Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels: int, filters: int, out_channels: int):
        super(Decoder, self).__init__()

        # Logger
        self.logger = logger.get_logger(self.__class__.__name__, logger.WARNING)

        # Layers
        self.layers = nn.ModuleDict({
            'block-1': nn.ModuleDict({
                'up-tail': nn.ModuleDict({}),
                'deconv-tail': nn.ModuleDict({})
            }),
            'block-2': nn.ModuleDict({
                'up-tail': nn.ModuleDict({}),
                'deconv-tail': nn.ModuleDict({})
            }),
            'skip-tail': nn.ModuleDict({}),
            'activation': nn.ReLU()
        })

        # Expander
        self.layers['expander'] = nn.Conv1d(in_channels, filters * 8, 3, stride = 1)
        
        # Block 1
        self.layers['block-1']['up-tail'] = nn.Upsample(scale_factor = 2, mode = 'linear')

        self.layers['block-1']['deconv-tail']['conv-t'] = nn.ConvTranspose1d(filters * 8, filters * 4, 2, stride = 2, padding = 0)
        self.layers['block-1']['deconv-tail']['conv'] = nn.Conv1d(filters * 4, filters * 8, 3, stride = 1)
        self.layers['block-1']['aggregator'] = nn.Conv1d(filters * 8 * 2, filters * 8, 3, stride = 1)
        
        # Block 2
        self.layers['block-2']['up-tail'] = nn.Upsample(scale_factor = 2, mode = 'linear')
        
        self.layers['block-2']['deconv-tail']['conv-t'] = nn.ConvTranspose1d(filters * 8, filters * 4, 2, stride = 2, padding = 0)
        self.layers['block-2']['deconv-tail']['conv'] = nn.Conv1d(filters * 4, filters * 8, 3, stride = 1)
        self.layers['block-2']['aggregator'] = nn.Conv1d(filters * 8 * 2, filters * 8, 3, stride = 1)
        
        # Skip Tail
        self.layers['skip-tail']['conv-t-1'] = nn.ConvTranspose1d(filters * 8, filters * 4, 2, stride = 2, padding = 0)
        self.layers['skip-tail']['conv-t-2'] = nn.ConvTranspose1d(filters * 4, filters * 8, 2, stride = 2, padding = 0)

        # Aggregator
        self.layers['aggregator'] = nn.Conv1d(filters * 8 * 2, out_channels, 3, stride = 1)

    # Aux function to apply pad 'sane' to strided convolutions
    def __pad(self, x, kernel, stride):
        L_in = x.size(-1)

        # Compute the total padding needed to ensure "same" output length
        L_out = math.ceil(L_in / stride)
        pad_needed = max(0, (L_out - 1) * stride + kernel - L_in)

        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left

        return F.pad(x, (pad_left, pad_right), mode = 'reflect')

    def forward(self, x):
        # Expander
        x = self.__pad(x, self.layers['expander'].kernel_size[0], self.layers['expander'].stride[0])
        expanded = self.layers['expander'](x)
        expanded = self.layers['activation'](expanded)
        
        # Block 1
        up_tail_b1 = self.layers['block-1']['up-tail'](expanded)

        deconv_tail_b1 = self.layers['block-1']['deconv-tail']['conv-t'](expanded)
        deconv_tail_b1 = self.layers['activation'](deconv_tail_b1)
        deconv_tail_b1 = self.__pad(deconv_tail_b1, self.layers['block-1']['deconv-tail']['conv'].kernel_size[0], self.layers['block-1']['deconv-tail']['conv'].stride[0])
        deconv_tail_b1 = self.layers['block-1']['deconv-tail']['conv'](deconv_tail_b1)
        deconv_tail_b1 = self.layers['activation'](deconv_tail_b1)

        # add_b1 = up_tail_b1 + deconv_tail_b1
        # add_b1 = self.__pad(add_b1, self.layers['block-1']['aggregator'].kernel_size[0], self.layers['block-1']['aggregator'].stride[0])
        concat_b1 = torch.cat([up_tail_b1, deconv_tail_b1], dim=1)
        concat_b1 = self.__pad(concat_b1, self.layers['block-1']['aggregator'].kernel_size[0], self.layers['block-1']['aggregator'].stride[0])
        
        # aggregator_b1 = self.layers['block-1']['aggregator'](add_b1)
        aggregator_b1 = self.layers['block-1']['aggregator'](concat_b1)
        aggregator_b1 = self.layers['activation'](aggregator_b1)
        
        self.logger.debug(f'Aggregator 1 - {aggregator_b1.shape}')

        
        # Block 2
        up_tail_b2 = self.layers['block-2']['up-tail'](aggregator_b1)


        deconv_tail_b2 = self.layers['block-2']['deconv-tail']['conv-t'](aggregator_b1)
        deconv_tail_b2 = self.layers['activation'](deconv_tail_b2)
        deconv_tail_b2 = self.__pad(deconv_tail_b2, self.layers['block-2']['deconv-tail']['conv'].kernel_size[0], self.layers['block-2']['deconv-tail']['conv'].stride[0])
        deconv_tail_b2 = self.layers['block-2']['deconv-tail']['conv'](deconv_tail_b2)
        deconv_tail_b2 = self.layers['activation'](deconv_tail_b2)

        # add_b2 = up_tail_b2 + deconv_tail_b2
        # add_b2 = self.__pad(add_b2, self.layers['block-2']['aggregator'].kernel_size[0], self.layers['block-2']['aggregator'].stride[0])
        concat_b2 = torch.cat([up_tail_b2, deconv_tail_b2], dim=1)
        concat_b2 = self.__pad(concat_b2, self.layers['block-2']['aggregator'].kernel_size[0], self.layers['block-2']['aggregator'].stride[0])
        
        # aggregator_b2 = self.layers['block-2']['aggregator'](add_b2)
        aggregator_b2 = self.layers['block-2']['aggregator'](concat_b2)
        aggregator_b2 = self.layers['activation'](aggregator_b2)


        self.logger.debug(f'Aggregator 2 - {aggregator_b2.shape}')
        
        # Skip Tail
        skip_tail = self.layers['skip-tail']['conv-t-1'](expanded)
        skip_tail = self.layers['activation'](skip_tail)
        skip_tail = self.layers['skip-tail']['conv-t-2'](skip_tail)
        skip_tail = self.layers['activation'](skip_tail)

        self.logger.debug(f'Skip Tail - {skip_tail.shape}')

        # Aggregator
        concat = torch.cat([aggregator_b2, skip_tail], dim=1)
        concat = self.__pad(concat, self.layers['aggregator'].kernel_size[0], self.layers['aggregator'].stride[0])
        out = self.layers['aggregator'](concat)
        
        return out



## AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self, in_channels: int, filters: int, embedding_channels: int):
        super(AutoEncoder, self).__init__()
        
        # Logger
        self.logger = logger.get_logger(self.__class__.__name__, logger.INFO)

        ## Compression ratio - INFORMATIVE
        self.compression_ratio = 4.0 * (float(in_channels) / float(embedding_channels))
        self.logger.info(f'Compression ratio: {self.compression_ratio:.2f}:1')

        # Components
        self.encoder = Encoder(in_channels, filters, embedding_channels)
        self.decoder = Decoder(embedding_channels, filters, in_channels)

    def forward(self, x):
        self.logger.debug(f'Input: {x.shape}')

        # Encode
        embedding = self.encoder(x)
        self.logger.debug(f'Embedding: {embedding.shape}')
        
        # Decode
        out = self.decoder(embedding)

        return out
