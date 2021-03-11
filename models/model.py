import torch
import torch.nn as nn
import torch.nn.functional as F

# various masking
from utils.masking import TriangularCausalMask, ProbMask
# encoder blocks
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack

# decoder blocks
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', data='ETTh1', freq='h', activation='gelu', 
                output_attention=False, distil=True, device=torch.device('cuda')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.e_layers = e_layers
        self.d_layers = d_layers

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, seq_len, d_model, embed, freq, dropout, data)
        self.dec_embedding = DataEmbedding(dec_in, label_len+out_len, d_model, embed, freq, dropout, data)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        if e_layers > 0:
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor,
                                            attention_dropout=dropout,
                                            output_attention=output_attention), 
                                            d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    # stacking multiple layers
                    ) for l in range(e_layers)
                ],
                [
                    ConvLayer(
                        d_model
                    # stacking multiple layers
                    ) for l in range(e_layers-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) if attn == 'prob' else \
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation), num_layers=e_layers, norm=nn.LayerNorm(d_model))
        else:
            self.encoder = nn.Identity()

        # Decoder
        if d_layers > 0:
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(ProbAttention(True, factor,
                                                    attention_dropout=dropout,
                                                    output_attention=False), 
                                                    d_model, n_heads),
                        AttentionLayer(FullAttention(False, factor,
                                                    attention_dropout=dropout,
                                                    output_attention=False), 
                                                    d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for l in range(d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            ) if attn == 'prob' else \
            nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, activation), num_layers=d_layers, norm=nn.LayerNorm(d_model))
        else:
            self.decoder = nn.Identity()

        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if self.e_layers > 0:
            if self.attn == 'prob':
                enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
            else:
                enc_out = enc_out.permute(1, 0, 2)
                enc_out = self.encoder(enc_out, mask=enc_self_mask)
                enc_out = enc_out.permute(1, 0, 2)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        if self.d_layers > 0:
            if self.attn == 'prob':
                dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            else:
                dec_out = dec_out.permute(1, 0, 2)
                enc_out = enc_out.permute(1, 0, 2)
            #  if dec_self_mask is not None:
            #      print('dec_self_mask shape: ', dec_self_mask.shape)
                dec_out = self.decoder(dec_out, enc_out, tgt_mask=enc_self_mask, memory_mask=dec_self_mask)
                dec_out = dec_out.permute(1, 0, 2)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        stacks = list(range(e_layers, 2, -1)) # you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor,
                                                 attention_dropout=dropout,
                                                 output_attention=False), 
                                                 d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor,
                                                 attention_dropout=dropout,
                                                 output_attention=False), 
                                                 d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                # stacking multiple layers
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #  print('x_enc: ', x_enc)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        #  print('enc_embed: ', enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        #  print('encoder: ', enc_out)

        #  print('x_dec: ', x_dec)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        #  print('dec_embed: ', dec_out)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        #  print('decoder: ', dec_out)
        dec_out = self.projection(dec_out)
        #  print('final output: ', dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
