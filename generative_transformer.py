import torch
import torch.nn as nn
import math


class PositionalEncoding( nn.Module ):
    def __init__( self, d_model, dropout=0.1, max_len=5000 ):
        super().__init__()
        self.dropout = nn.Dropout( p=dropout )
        position = torch.arange( max_len ).unsqueeze( 1 )
        div_term = torch.exp( torch.arange( 0, d_model, 2 ) * \
                              ( -math.log( 10000.0 ) / d_model ) )
        pe = torch.zeros( max_len, 1, d_model )
        pe[ :, 0, 0::2 ] = torch.sin( position * div_term )
        pe[ :, 0, 1::2 ] = torch.cos( position * div_term )
        self.register_buffer( 'pe', pe )

    def forward( self, x ):
        # shape: ( Batch, Seq, Dim )
        x = x + self.pe[ :x.size(1), : ].transpose( 0, 1 )
        return self.dropout( x )


class GenerativeTransformer( nn.Module ):
    def __init__( self, vocab_size, d_model, nheads, num_layers, dropout=0.1 ):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model

        self.embedding = nn.Embedding( vocab_size, d_model )
        self.pos_encoder = PositionalEncoding( d_model, dropout )

        encoder_layers = nn.TransformerEncoderLayer( d_model, 
                                                     nhead, 
                                                     dim_feedforward=d_model*4, 
                                                     dropout=dropout, 
                                                     batch_first=True )
        self.transformer_encoder = nn.TransformerEncoder( encoder_layers, 
                                                          num_layers )
        self.decoder = nn.Linear( d_model, vocab_size )
        self.init_weights()

    def init_weights( self ):
        initrange = 0.1
        self.enbedding.weight.data.uniform_(
            -initrange, initrange )
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(
            -initrange, initrange )

    def _generate_square_subsequent_mask( self, size ):
        """Causal mask"""
        mask = ( torch.triu( torch.ones( size, size ) ) == 1 ) \
                    .transpose( 0, 1 )
        mask = mask.float().masked_fill( 
                                mask == 0, float('-inf') ) \
                           .masked_fill(
                                mask == 1, float( 0.0 ) )
        return mask

    def forward( self, x, mask=None ):
        x = self.embedding( x ) * math.sqrt( self.d_model )
        x = self.pos_encoder( x )
        
        if mask is None:
            device = x.device
            seq_len = x.size( 1 )
            mask = self._generate_square_subsequent_mask(
                        seq_len ).to( device )
            output = self.transformer_encoder( x, mask=mask,
                                               is_causal=True )
            output = self.decoder( output )
            return output
 
