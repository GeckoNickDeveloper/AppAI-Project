# Imports
import torch.nn as nn

# Network definition
## Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.config = {
            'convolutions': {    
                'in-channels': [
                    in_channels,
                    in_channels * 2 + 16,
                    (in_channels * 2 + 16) * 2 + 32
                ],
                'out-channels': [
                    16,
                    32,
                    in_channels
                ],
                'kernels': [
                    7,
                    3,
                    1
                ],
                'strides': [
                    2,
                    2,
                    1
                ]
            },
            'poolings': {
                'max-pool': {
                    'kernels': [
                        7,
                        3
                    ],
                    'strides': [
                        2,
                        2
                    ]
                },
                'avg-pool': {
                    'kernels': [
                        7,
                        3
                    ],
                    'strides': [
                        2,
                        2
                    ]
                },
                'lp-pool': {
                    'p': 5,
                    'kernel': 3,
                    'stride': 5
                },
            }
        }

        # Block 1
        self.max_pool_b1 = nn.MaxPool1d(
                self.config['poolings']['max-pool']['kernels'][0],
                stride = self.config['poolings']['max-pool']['kernels'][0],
                padding = 0)

        self.avg_pool_b1 = nn.AvgPool1d(
                self.config['poolings']['avg-pool']['kernels'][0],
                stride = self.config['poolings']['avg-pool']['kernels'][0],
                padding = 0)
        
        self.conv_b1 = nn.Conv1d(
                self.config['convolutions']['in-channels'][0],
                self.config['convolutions']['out-channels'][0]
                self.config['convolutions']['kernels'][0],
                stride = self.config['convolutions']['strides'][0],
                padding = 'same',
                padding_mode = 'reflect')
        


        # Block 2
        self.max_pool_b2 = nn.MaxPool1d(
                self.config['poolings']['max-pool']['kernels'][1],
                stride = self.config['poolings']['max-pool']['kernels'][1],
                padding = 0)
        
        self.avg_pool_b2 = nn.AvgPool1d(
                self.config['poolings']['avg-pool']['kernels'][1],
                stride = self.config['poolings']['avg-pool']['kernels'][1],
                padding = 0)
        
        self.conv_b2 = nn.Conv1d(
                self.config['convolutions']['in-channels'][1],
                self.config['convolutions']['out-channels'][1]
                self.config['convolutions']['kernels'][1],
                stride = self.config['convolutions']['strides'][1],
                padding = 'same',
                padding_mode = 'reflect')



        # Block 3
        self.conv_b3 = nn.Conv1d(
                self.config['convolutions']['in-channels'][2],
                self.config['convolutions']['out-channels'][2]
                self.config['convolutions']['kernels'][2],
                stride = self.config['convolutions']['strides'][2],
                padding = 'same',
                padding_mode = 'reflect')

        self.lp_pool_b3 = nn.LPPool1d(
                self.config['poolings']['lp-pool']['p'],
                self.config['poolings']['lp-pool']['kernel'],
                stride = self.config['poolings']['lp-pool']['stride'])



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
    def __init__(self, in_channels):
        super(Decoder, self).__init__()

        self.config = {
            'deconvolutions': {
                'in-channels': [
                    in_channels,
                    in_channels + 16,
                ],
                'out-channels': [
                    16,
                    32,
                ],
                'kernels': [
                    4,
                    2,
                ],
                'strides': [
                    4,
                    2,
                ]
            },

            'convolutions': {
                'in-channels': [
                    in_channels + 16 + 32,
                    in_channels,
                ],
                'out-channels': [
                    in_channels,
                    in_channels,
                ],
                'kernels': [
                    3,
                    5,
                ],
                'strides': [
                    1,
                    1,
                ]

            },
            
            'upsamples': {
                'scale-factors': [
                    4,
                    5,
                    20
                ]
            }
        }



        # Block 1
        self.up_b1 = nn.Upsample(
                scale_factor = self.config['upsamples']['scale-factors'][0],
                mode = 'linear')

        self.convt1_b1 = nn.ConvTranspose1d(
                self.config['deconvolutions']['in-channels'][0],
                self.config['deconvolutions']['out-channels'][0],
                self.config['deconvolutions']['kernels'][0],
                stride = self.config['deconvolutions']['strides'][0],
                padding = 0,
                output_padding = 0,
                padding_mode = 'zeros') # TODO: Tune params
        


        # Block 2
        self.up_b2 = nn.Upsample(
                scale_factor = self.config['upsamples']['scale-factors'][1],
                mode = 'linear')
        
        self.convt1_b2 = nn.ConvTranspose1d(
                self.config['deconvolutions']['in-channels'][1],
                self.config['deconvolutions']['out-channels'][1],
                self.config['deconvolutions']['kernels'][1],
                stride = self.config['deconvolutions']['strides'][1],
                padding = 0,
                output_padding = 0,
                padding_mode = 'zeros') # TODO: Tune params
        


        # Block 3
        self.embedding_up = nn.Upsample(
                scale_factor = self.config['upsamples']['scale-factors'][2],
                mode = 'linear')
        
        self.conv1_b3 = nn.Conv1d(
                self.config['convolutions']['in-channels'][0],
                self.config['convolutions']['out-channels'][0],
                self.config['convolutions']['kernels'][0],
                stride = self.config['convolutions']['strides'][0],
                padding = 0,
                padding_mode = 'zeros') # TODO: Tune params

        self.conv2_b3 = nn.Conv1d(
                self.config['convolutions']['in-channels'][1],
                self.config['convolutions']['out-channels'][1],
                self.config['convolutions']['kernels'][1],
                stride = self.config['convolutions']['strides'][1],
                padding = 0,
                padding_mode = 'zeros') # TODO: Tune params



    def forward(self, x):
        # Block 1
        up_b1 = self.up_b1(x)
        ct1_b1 = self.convt1_b1(x)

        concat_b1 = torch.cat([up_b1, ct1_b1], dim=1)

        # Block 2
        up_b2 = self.up_b1(concat_b1)
        ct1_b2 = self.convt1_b2(concat_b2)

        concat_b2 = torch.cat([up_b2, ct1_b2], dim=1) 

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
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # Components
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        # Encode
        embedding = self.encoder(x)
        
        # Decode
        out = self.decoder(embedding)

        return out
