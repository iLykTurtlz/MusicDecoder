import torch
import torch.nn as nn
import pspython

class MIDITransformer( pspython.neural.DeepModel ):
    _create_model( self,
                   d_model,
                   nhead,
                   num_decoder_layers,
                   dim_feedforward,
                   dropout ):
        return nn.Transformer
                
