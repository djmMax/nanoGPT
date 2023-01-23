"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class IOProcessingAttention(nn.Module):
    # TODO check if split this class in Output and Input, effect on performance and code readability
    # TODO input/output dim can be different than latent dim (here is config.n_embd)
    def __init__(self, config, type='input', io_dim=None, mult_head_dim=None):
        super().__init__()
        self.is_encoding = type == 'input'
        self.is_decoding = type == 'ouput'
        self.is_process  = type == 'process' # same as basic self attention
        self.io_dim = io_dim if io_dim else config.n_embd
        self.mult_head_dim = mult_head_dim if mult_head_dim else config.n_embd
        
        assert self.mult_head_dim % config.n_head == 0

        if self.is_process:
            self.attn_l = nn.Linear(config.n_embd, 3 * self.mult_head_dim)
        else:
            # key, query, value projections for all ouput heads, but in a batch
            self.attn_c = nn.Linear(io_dim, 2 * self.mult_head_dim)
            # key, value projections for latent heads
            self.attn_l = nn.Linear(config.n_embd, 2 * self.mult_head_dim)
            self.attn_query = nn.Linear(config.n_embd if self.is_encoding else self.io_dim, self.mult_head_dim) # HACK so that with attn_ use we get k and v -> 2 * config.n_embd

        # output projection
        self.c_proj = nn.Linear(mult_head_dim, io_dim)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if self.is_decoding:
            # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
            #                             .view(1, 1, config.block_size, config.block_size))
            # this will assume blocksize == max_number_latent
            # causal mask to ensure that attention is only applied to the left in the ouput sequence
            # TODO check if torch.tril(..., diagonal=...) is correctly used
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size*2), diagonal=config.block_size-1)
                                        .view(1, 1, config.block_size, config.block_size*2))

        self.block_size = config.block_size
        self.latent_size = config.block_size # TODO make latent size ajustable
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, l, x=None):
        # if x is None:
        #     B, L, C = l.size()
        #     T = L
        # else:
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (mult_head_dim original code was n_embd)
        _, L, _ = l.size() # latent input

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # TODO compute only q_l xor q_c since we are just using one of them at ounce
        if self.is_process:
            q, k ,v = self.attn_l(l).split(self.mult_head_dim, dim=2)
        else:
            k_l ,v_l = self.attn_l(l).split(self.mult_head_dim, dim=2)
            k_c ,v_c  = self.attn_c(x).split(self.mult_head_dim, dim=2)
        
            # TODO check if conditiontal statement affect performance (eg. compile), if yes split this class
            # TODO check if pytorch compile does similar work to FlashAttention, recurisve TODO read FlashAttention + understand GPU/Auto-diff-graph
            # TODO check if we can freeze conditiontal statement in case they affect performance (eg. functionnal programming with fixed argument)
            # in decoding the output is the sequence
            if self.is_decoding:
                k = torch.cat((k_l, k_c)) # TODO check if torch compiler will combine k_l, k_c effeceintly without using more memmory 
                q = self.attn_query(x) # query from the context for the output # q = q_c
                v = torch.cat((v_l, v_c))
            
            # in enconding the output is the latent 
            if self.is_encoding:
                k = torch.cat((k_c, k_l)) # is not k_l, k_c because permute them make the ouput the latent
                q = self.attn_query(l) # query from the latent for the output # q = q_l
                v = torch.cat((v_c, v_l))

        k = k.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, L+T, hs) or (B, nh, T+L, hs) or (B, nh, L, hs)
        q = q.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh,   T, hs) or (B, nh,   L, hs) or (B, nh, L, hs)
        v = v.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, L+T, hs) or (B, nh, T+L, hs) or (B, nh, L, hs)

        # self-attention + partial causal attention on x; Self-attend: (B, nh, T, hs) x (B, nh, hs, L+T) -> (B, nh, T, L+T) for decoding
        # self-attention + partial causal attention on l; Self-attend: (B, nh, L, hs) x (B, nh, hs, T+L) -> (B, nh, T, T+L) for encoding
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.is_decoding: 
            # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            begin = self.block_size - L # TODO check if this is a bottleneck
            end   = begin + L + T
            att   = att.masked_fill(self.bias[:, :, begin:end, begin:end] == 0, float('-inf')) # (B, nh, T, L+T) # TODO double check this mask 
            # IMPROVEMENT we may have variable number of latent # self.bias[:,:,T-L:L+T,T-L:L+T]
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v #    (B, nh, T, L+T) x (B, nh, L+T, hs) -> (B, nh, T, hs) # for decoding
                    # or (B, nh, L, T+L) x (B, nh, T+L, hs) -> (B, nh, L, hs) # for encoding
        
        y = y.transpose(1, 2).contiguous() # re-assemble all head outputs side by side
        y = y.view(B, L, C) if self.is_encoding else y.view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class InputBlock(nn.Module):
    pass
class OutputBlock(nn.Module):
    pass
class ProcessBlock(nn.Module):
    pass
class BlockComposes(nn.Module):
    def forward(self,x):
        # NOTE compose block base on config
        # NOTE we may dynamic forward pass
        pass

class CausalEvaluation:
    pass
class IndexerEvaluation:
    pass
class Noise:
    pass

cfg = dict()
# d = lambda **kwarg: dict(**kwarg)
def d(model, **kwarg):
    kwarg['model'] = model
    return dict(**kwarg) # rename dict function

config = dict(
    ## input
    text_embedding = dict(vocab_size=256, io_dim=32),
    init_latent = dict(model=Noise, number_of_latent=128, dim=512), # or  dict(vocab_size=25, dim=512),

    ## repeat this
    indexer = dict(
        model=        [OutputBlock], # or IndexerBlock
        io='1. latent -> index_schema'),
    text_reader = d(  [InputBlock,ProcessBlock], positional_emb='alibi',
        io='2. text_embedding, latent -> latent',
        random='on/off'),
    latent_reader =d( InputBlock, io_dim='cfg.latent_dim', # io_dim = latent_dim
        io='3. text_reader.latent, prev_latent, context.latent -> latent'), # random stop gradient prev_latent
    processor =d(     [ProcessBlock]*4, io_dim=cfg.latent_dim,
        io='4. latent_reader -> latent'), # random stop gradient latent
    text_writer =d(   [OutputBlock]*2,
        io='5. processor.latent -> output_text_emb'),
    
    ### intermediate evaluation
    eval_ouput = d(
        model=CausalEvaluation, # with confidence probe
        io='eval. ground_throught,output_text_emb -> loss'),
    eval_indexer = d(
        model = IndexerEvaluation,
        io='eval. text_reader.attn_map, latent_reader.attn_map, index_schema -> loss'),

    ### train to use old nkowledge
    save_for_later = 'add_data(prev_latent, batch.idx)',
    ###### we can make we provide close to context data (learn to reuse latent), or even close to context aggregated concept (learn to synthitise)

    ### training
    loss = 'eval_ouput.loss + eval_indexer.loss * 0.1',
    prev_latent = 'prev_latent.append(latent)',

    ### loggin need to consider randomizaton during training, to how performance evolduring training
    ###### or performing in many way conisidering ablation/parameter search

    ### after 1-4 repeat
    text_token = 'text_token.append(data.get(token=32))', # shoul be nice to have variable number of added token
    # witch means redo : text_token -> text_embedding

    #general config
    cfg=dict(
        _input_lenght=300,
        _output_lenght=50,
        _io_dim=32,
        _prev_latent_history=5,
        _latent_size=20,
        _latent_dim=512,
        _tokenizer='char_tokenizer'
    )
)


'''
# another synthax

## input
text_token -> text_embedding

## repeat this
indexer = OutputBlock
    latent -> index_schema
text_reader = InputBlock, ProcessBlock # random on/off
    text_embedding, latent -> latent
latent_reader = InputBlock, io_dim=cfg.latent_dim  # random on/off
    text_reader.latent, prev_latent -> latent
processor = ProcessBlock*4
    latent_reader -> latent
text_writer = OutputBlock*2            # maybe random on/off
    processor.latent -> output_text_emb

### intermediate evaluation
eval_ouput = CausalEvaluation # with confident probe
    ground_throught, output_text_emb -> loss
eval_indexer = IndexerEvaluation
    text_reader.attn_map, latent_reader.attn_map, index_schema -> loss
loss = eval_ouput.loss + eval_indexer.loss * 0.1

prev_latent.append(latent)

### after 1-4 repeat
text_token.append(data.get(token=32)) # shoul be nice to have variable number of added token
text_token -> text_embedding


## default param
_latent_dim  = 256
_latent_size = 32
_max_token_output = 32
_input_token = 256

_tokenizer   = char_tokenizer # will set _text_embed = 128


NOTE: we can read ounce, by combining text_reader + latent_reader, by making the InputBlock work on input with diffreent dimension
k1, v1 = slef.attn_text(text_embed)         # (T, Emb_dim)    -> (T, head_dim)
k2, v2 = slef.attn_prev_latent(prev_latent) # (P, Latent_dim) -> (P, head_dim)
... 
q, k_l, v_l = slef.attn_latent(latent) # (L, Latent_dim) -> (L, head_dim)

k = torch.cat(k1, k2, ..., k_l) # (T + P + L, head_dim)
v = torch.cat(v1, v2, ..., v_l) # (T + P + L, head_dim)

attn_map = attn(q, k.T) # (L, head_dim) x (head_dim,T + P + L) -> (L, T + P + L)
y = attn_map @ v        # (L, T + P + L) x (T + P + L, head_dim) -> (L, head_dim)

# IMPROVEMENT, head_dim of v can be independent than k, that we v.head_dim=Latent_dim for the output and k can be smaller. but we may have issue with multihead!
# k1, v1, k2, v2,... can be implemented via sub class with their forward() that we will use to read those input
'''


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # we can override the dropout rate
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']
        # block_size is always 1024 for GPT model checkpoints
        # if one wants a lower block_size it has to be done through model surgery
        # later, by calling crop_block_size()

        # create a from-scratch initialized minGPT model
        config = GPTConfig(block_size=1024, **config_args)
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
