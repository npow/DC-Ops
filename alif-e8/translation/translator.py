"""Fixed-size Marian encoder/decoder using the pretrained OPUS-MT weights."""
import math

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.q = attention.q_proj
        self.k = attention.k_proj
        self.v = attention.v_proj
        self.out = attention.out_proj
        self.heads = attention.num_heads
        self.dim = attention.head_dim

    def forward(self, hidden, context, mask):
        q = self.q(hidden).reshape(1, -1, self.heads, self.dim).transpose(1, 2)
        k = self.k(context).reshape(1, -1, self.heads, self.dim).transpose(1, 2)
        v = self.v(context).reshape(1, -1, self.heads, self.dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim)
        weights = torch.softmax(scores + mask, dim=-1)
        result = torch.matmul(weights, v).transpose(1, 2).reshape(1, -1, self.heads*self.dim)
        return self.out(result)


class EncoderLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.attention = Attention(layer.self_attn)
        self.norm1 = layer.self_attn_layer_norm
        self.norm2 = layer.final_layer_norm
        self.fc1 = layer.fc1
        self.fc2 = layer.fc2

    def forward(self, hidden, mask):
        hidden = self.norm1(hidden + self.attention(hidden, hidden, mask))
        return self.norm2(hidden + self.fc2(torch.relu(self.fc1(hidden))))


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(layer) for layer in model.model.encoder.layers)

    def forward(self, embeddings, mask):
        for layer in self.layers:
            embeddings = layer(embeddings, mask)
        return embeddings


class DecoderLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.attention = Attention(layer.self_attn)
        self.cross_attention = Attention(layer.encoder_attn)
        self.norm1 = layer.self_attn_layer_norm
        self.norm2 = layer.encoder_attn_layer_norm
        self.norm3 = layer.final_layer_norm
        self.fc1 = layer.fc1
        self.fc2 = layer.fc2

    def forward(self, hidden, encoded, self_mask, encoder_mask):
        hidden = self.norm1(hidden + self.attention(hidden, hidden, self_mask))
        hidden = self.norm2(hidden + self.cross_attention(hidden, encoded, encoder_mask))
        return self.norm3(hidden + self.fc2(torch.relu(self.fc1(hidden))))


class Decoder(nn.Module):
    def __init__(self, model, length):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(layer) for layer in model.model.decoder.layers)
        self.head = model.lm_head
        self.register_buffer("bias", model.final_logits_bias)
        self.register_buffer("causal_mask", torch.triu(torch.full((1, 1, length, length), -100.0), diagonal=1))

    def forward(self, embeddings, encoded, encoder_mask, last_position):
        for layer in self.layers:
            embeddings = layer(embeddings, encoded, self.causal_mask, encoder_mask)
        hidden = (embeddings * last_position).sum(dim=1)
        return self.head(hidden) + self.bias


class Translator:
    def __init__(self, model, tokenizer, length=32):
        self.model = model
        self.tokenizer = tokenizer
        self.length = length
        self.encoder = Encoder(model).eval()
        self.decoder = Decoder(model, length).eval()

    def embed(self, ids, decoder=False):
        component = self.model.model.decoder if decoder else self.model.model.encoder
        return component.embed_tokens(ids)*component.embed_scale + component.embed_positions(ids.shape)

    def encode_inputs(self, text):
        batch = self.tokenizer(text, return_tensors="pt")
        if batch.input_ids.shape[1] > self.length:
            raise ValueError(f"Input exceeds {self.length} tokens: {text}")
        ids = torch.full((1, self.length), self.model.config.pad_token_id, dtype=torch.long)
        ids[:, :batch.input_ids.shape[1]] = batch.input_ids
        mask = (ids == self.model.config.pad_token_id).float()[:, None, None, :]*-100.0
        return self.embed(ids), mask

    def decoder_inputs(self, tokens, encoded, mask):
        ids = torch.full((1, self.length), self.model.config.pad_token_id, dtype=torch.long)
        ids[0, :len(tokens)] = torch.tensor(tokens)
        last = torch.zeros((1, self.length, 1))
        last[0, len(tokens)-1, 0] = 1
        return self.embed(ids, decoder=True), encoded, mask, last

    @torch.no_grad()
    def translate(self, text, encoder=None, decoder=None, collect=False):
        enc_args = self.encode_inputs(text)
        encoded = (encoder if encoder is not None else self.encoder)(*enc_args)
        tokens = [self.model.config.decoder_start_token_id]
        traces = []
        for _ in range(self.length-1):
            args = self.decoder_inputs(tokens, encoded, enc_args[1])
            if collect:
                traces.append(tuple(x.detach().clone() for x in args))
            logits = (decoder if decoder is not None else self.decoder)(*args)
            logits[:, self.model.config.pad_token_id] = -float("inf")
            token = int(logits.argmax(-1).item())
            tokens.append(token)
            if token == self.model.config.eos_token_id:
                break
        return {"text": self.tokenizer.decode(tokens[1:], skip_special_tokens=True),
                "tokens": tokens[1:], "finished": tokens[-1] == self.model.config.eos_token_id,
                "encoder_inputs": enc_args if collect else None, "decoder_inputs": traces}
