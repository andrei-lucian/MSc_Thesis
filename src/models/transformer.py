# transformer_iwslt.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Positional Encoding (sinusoidal)
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                     # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register as buffer: not a parameter, moves with .to(device)
        self.register_buffer("pe", pe.unsqueeze(0))            # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# -------------------------------
# Encoder/Decoder Sublayers
# (Post-Norm as in Vaswani)
# -------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()  # <-- nn.ReLU module
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(src, src, src,
                                     attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout_attn(attn_out))

        ff = self.linear2(self.relu(self.linear1(src)))  # <-- changed
        src = self.norm2(src + self.dropout_ff(ff))
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()  # <-- nn.ReLU module
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        sa_out, _ = self.self_attn(tgt, tgt, tgt,
                                   attn_mask=tgt_mask,
                                   key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout_self(sa_out))

        ca_out, _ = self.cross_attn(tgt, memory, memory,
                                    attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout_cross(ca_out))

        ff = self.linear2(self.relu(self.linear1(tgt)))  # <-- changed
        tgt = self.norm3(tgt + self.dropout_ff(ff))
        return tgt


# -------------------------------
# Encoder & Decoder Stacks
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, d_ff, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        for layer in self.layers:
            x = layer(x, src_mask, src_key_padding_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, d_ff, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        for layer in self.layers:
            x = layer(x, memory,
                      tgt_mask, memory_mask,
                      tgt_key_padding_mask, memory_key_padding_mask)
        return x


# -------------------------------
# Full Seq2Seq Transformer
# -------------------------------
class Seq2SeqTransformer(nn.Module):
    """
    Encoder–Decoder Transformer (Vaswani et al., 2017)
    - 6 encoder layers, 6 decoder layers (configurable)
    - 8 heads (configurable)
    - d_ff = 4 * d_model
    - dropout defaults to 0.0 per your setup (label smoothing handled in the loss)
    - optional decoder weight tying with output projection

    Inputs:
      src: [B, S]  int token ids
      tgt: [B, T]  int token ids (teacher-forced, right-shifted)
    Outputs:
      logits: [B, T, tgt_vocab] (unnormalized)
    """
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dropout: float = 0.0,
                 tie_decoder_embeddings: bool = True,
                 max_len: int = 5000):
        super().__init__()

        d_ff = 4 * d_model

        # Embeddings
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        # Stacks
        self.encoder = Encoder(d_model, nhead, num_encoder_layers, d_ff, dropout)
        self.decoder = Decoder(d_model, nhead, num_decoder_layers, d_ff, dropout)

        # Generator (projection to vocab)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying (common in MT; also used in fairseq configs)
        if tie_decoder_embeddings:
            self.generator.weight = self.tgt_tok_emb.weight

        # Initialization (Vaswani-style)
        self._reset_parameters(d_model)

    def _reset_parameters(self, d_model: int):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Embed scale (like fairseq)
        self.embed_scale = math.sqrt(d_model)

    # ---- Mask helpers (plug-and-play) ----
    @staticmethod
    def generate_square_subsequent_mask(sz: int, device=None):
        # causal mask for decoder self-attn
        # shape [sz, sz], True means "mask out"
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask if device is None else mask.to(device)

    @staticmethod
    def make_padding_mask(seq: torch.Tensor, pad_idx: int):
        # seq: [B, L] -> mask: [B, L], True where pad
        return (seq == pad_idx)

    def encode(self, src_ids, src_key_padding_mask=None):
        # src_ids: [B, S] ints
        src = self.embed_scale * self.src_tok_emb(src_ids)      # [B, S, d_model]
        src = self.pos_enc(src)                                 # add positional
        memory = self.encoder(src, src_mask=None,
                              src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, tgt_ids, memory,
               tgt_mask=None,
               tgt_key_padding_mask=None,
               memory_key_padding_mask=None):
        # tgt_ids: [B, T] ints
        tgt = self.embed_scale * self.tgt_tok_emb(tgt_ids)      # [B, T, d_model]
        tgt = self.pos_enc(tgt)
        out = self.decoder(tgt, memory,
                           tgt_mask=tgt_mask,
                           memory_mask=None,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        return out

    def forward(self, src_ids, tgt_ids,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                tgt_is_causal: bool = True,
                pad_idx: int = None):
        """
        - If tgt_is_causal=True, apply a standard causal mask to decoder self-attn.
        - Padding masks should be Boolean with True at PAD positions.
        """
        B, T = tgt_ids.size()
        device = tgt_ids.device

        # Build causal mask if needed
        tgt_mask = None
        if tgt_is_causal:
            tgt_mask = self.generate_square_subsequent_mask(T, device=device)

        memory = self.encode(src_ids, src_key_padding_mask=src_key_padding_mask)
        dec_out = self.decode(tgt_ids, memory,
                              tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)
        logits = self.generator(dec_out)                         # [B, T, V_tgt]
        return logits


# -------------------------------
# Factory for your replication
# -------------------------------
def transformer_iwslt14(src_vocab_size: int,
                        tgt_vocab_size: int,
                        d_model: int = 512,
                        nhead: int = 8,
                        num_layers: int = 6,
                        dropout: float = 0.0,
                        tie_decoder_embeddings: bool = True):
    """
    Encoder–decoder Transformer per Vaswani et al. (2017) / fairseq Ott et al. (2019),
    scaled by d_model with d_ff = 4 * d_model, 6 layers, 8 heads.
    """
    return Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dropout=dropout,
        tie_decoder_embeddings=tie_decoder_embeddings
    )
