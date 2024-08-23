from transformers import (RobertaModel, PretrainedConfig, AutoModel, PreTrainedModel, MPNetModel)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput
from einops import rearrange, reduce, repeat
from torch.utils.checkpoint import checkpoint
from typing import Optional
from dataclasses import dataclass
import math
from functools import partial


@dataclass
class SimilarOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    similarity: torch.FloatTensor = None

@dataclass
class LUARSimilarOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

# RoBERTa Model
class RoBERTa(RobertaModel):

    def __init__(self, config):
        super().__init__(config)
        self.distance_metric = lambda x, y: 1 - F.cosine_similarity(x, y)
        self.margin = 0.5
    
    def mean_pooling(self, embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self,
               input_ids,
               attention_mask,
               utter_mask):
        # input_ids: [bsz, utters, tokens]
        # attention_mask: [bsz, utters, tokens]
        # utter_mask: [bsz, utters]
        bsz, utters, _ = input_ids.shape
        input_ids = input_ids.reshape(bsz*utters, -1)
        attention_mask = attention_mask.reshape(bsz*utters, -1)
        token_embeddings = super().forward(input_ids=input_ids, attention_mask=attention_mask)[0]
        # token_embeddings: [bsz*utters, tokens, d_model]
        utter_embeddings = self.mean_pooling(token_embeddings, attention_mask)
        utter_embeddings = utter_embeddings.reshape(bsz, utters, -1)
        # utter_embeddings: [bsz, utters, d_model]
        speaker_embeddings = self.mean_pooling(utter_embeddings, utter_mask)
        # speaker_embeddings: [bsz, d_model]
        return speaker_embeddings
    
    def forward(self,
                input_ids1,
                attention_mask1,
                utter_mask1,
                input_ids2,
                attention_mask2,
                utter_mask2,
                labels=None):
        speaker_embeddings1 = self.encode(input_ids1, attention_mask1, utter_mask1)
        speaker_embeddings2 = self.encode(input_ids2, attention_mask2, utter_mask2)
        loss = None
        distances = self.distance_metric(speaker_embeddings1, speaker_embeddings2)
        logits = 1 - distances
        if labels is not None:
            losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
            loss = losses.mean()

        return SimilarOutput(
            loss=loss,
            similarity=logits
        )

# SBERT Model
class SBERT(MPNetModel):

    def __init__(self, config):
        super().__init__(config)
        self.distance_metric = lambda x, y: 1 - F.cosine_similarity(x, y)
        self.margin = 0.5
    
    def mean_pooling(self, embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self,
               input_ids,
               attention_mask,
               utter_mask):
        # input_ids: [bsz, utters, tokens]
        # attention_mask: [bsz, utters, tokens]
        # utter_mask: [bsz, utters]
        bsz, utters, _ = input_ids.shape
        input_ids = input_ids.reshape(bsz*utters, -1)
        attention_mask = attention_mask.reshape(bsz*utters, -1)
        token_embeddings = super().forward(input_ids=input_ids, attention_mask=attention_mask)[0]
        # token_embeddings: [bsz*utters, tokens, d_model]
        utter_embeddings = self.mean_pooling(token_embeddings, attention_mask)
        utter_embeddings = utter_embeddings.reshape(bsz, utters, -1)
        # utter_embeddings: [bsz, utters, d_model]
        speaker_embeddings = self.mean_pooling(utter_embeddings, utter_mask)
        # speaker_embeddings: [bsz, d_model]
        return speaker_embeddings
    
    def forward(self,
                input_ids1,
                attention_mask1,
                utter_mask1,
                input_ids2,
                attention_mask2,
                utter_mask2,
                labels=None):
        speaker_embeddings1 = self.encode(input_ids1, attention_mask1, utter_mask1)
        speaker_embeddings2 = self.encode(input_ids2, attention_mask2, utter_mask2)
        loss = None
        distances = self.distance_metric(speaker_embeddings1, speaker_embeddings2)
        logits = 1 - distances
        if labels is not None:
            losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
            loss = losses.mean()

        return SimilarOutput(
            loss=loss,
            similarity=logits
        )

# STEL Model
class STEL(RobertaModel):

    def __init__(self, config):
        super().__init__(config)
        self.distance_metric = lambda x, y: 1 - F.cosine_similarity(x, y)
        self.margin = 0.5
 
    def mean_pooling(self, embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self,
               input_ids,
               attention_mask,
               utter_mask):
        # input_ids: [bsz, utters, tokens]
        # attention_mask: [bsz, utters, tokens]
        # utter_mask: [bsz, utters]
        bsz, utters, _ = input_ids.shape
        input_ids = input_ids.reshape(bsz*utters, -1)
        attention_mask = attention_mask.reshape(bsz*utters, -1)
        token_embeddings = super().forward(input_ids=input_ids, attention_mask=attention_mask)[0]
        # token_embeddings: [bsz*utters, tokens, d_model]
        utter_embeddings = self.mean_pooling(token_embeddings, attention_mask)
        utter_embeddings = utter_embeddings.reshape(bsz, utters, -1)
        # utter_embeddings: [bsz, utters, d_model]
        speaker_embeddings = self.mean_pooling(utter_embeddings, utter_mask)
        # speaker_embeddings: [bsz, d_model]
        return speaker_embeddings
    
    def forward(self,
                input_ids1,
                attention_mask1,
                utter_mask1,
                input_ids2,
                attention_mask2,
                utter_mask2,
                labels=None):
        speaker_embeddings1 = self.encode(input_ids1, attention_mask1, utter_mask1)
        speaker_embeddings2 = self.encode(input_ids2, attention_mask2, utter_mask2)
        loss = None
        distances = self.distance_metric(speaker_embeddings1, speaker_embeddings2)
        logits = 1 - distances
        if labels is not None:
            losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
            loss = losses.mean()

        return SimilarOutput(
            loss=loss,
            similarity=logits
        )

# LUAR Model
class LUARConfig(PretrainedConfig):
    model_type = "LUAR"
    
    def __init__(self,
        embedding_size: int = 512,
        use_memory_efficient_attention=False,
        q_bucket_size=512,
        k_bucket_size=1024,
        **kwargs,
    ):
        self.embedding_size = embedding_size
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size
        super().__init__(**kwargs)

# LUAR Model
def exists(val):
    return val is not None

def summarize_qkv_chunk(
    q, k, v, 
    mask
):
    """Dot-Product Attention for a chunk of queries, keys, and values.
    """
    weight = torch.einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(mask):
        # HuggingFace masks have to be added:
        weight += mask

    weight_max = weight.amax(dim = -1, keepdim = True).detach()
    weight = weight - weight_max

    exp_weight = weight.exp()
    weighted_value = torch.einsum('b h i j, b h j d -> b h i d', exp_weight, v)

    return exp_weight.sum(dim = -1), weighted_value, rearrange(weight_max, '... 1 -> ...')

checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)

def memory_efficient_attention(
    q, k, v,
    mask = None,
    q_bucket_size = 512,
    k_bucket_size = 1024,
    eps = 1e-8
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # function
    needs_backwards = q.requires_grad or k.requires_grad or v.requires_grad
    summarize_qkv_fn = checkpointed_summarize_qkv_chunk if needs_backwards else summarize_qkv_chunk

    # chunk all the inputs
    q_chunks = q.split(q_bucket_size, dim = -2)
    k_chunks = k.split(k_bucket_size, dim = -2)
    v_chunks = v.split(k_bucket_size, dim = -2)
    mask_chunks = mask.split(k_bucket_size, dim = -1) if exists(mask) else ((None,) * len(k_chunks))

    # loop through all chunks and accumulate
    out = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights = []
        weighted_values = []
        weight_maxes = []
        
        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):

            exp_weight_chunk, weighted_value_chunk, weight_max_chunk = summarize_qkv_fn(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
            )

            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)
            weight_maxes.append(weight_max_chunk)

        exp_weights = torch.stack(exp_weights, dim = -1)
        weighted_values = torch.stack(weighted_values, dim = -1)
        weight_maxes = torch.stack(weight_maxes, dim = -1)

        global_max = weight_maxes.amax(dim = -1, keepdim = True)
        renorm_factor = (weight_maxes - global_max).exp().detach()

        exp_weights = exp_weights * renorm_factor
        weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')

        all_values = weighted_values.sum(dim = -1)
        all_weights = exp_weights.sum(dim = -1)

        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        out.append(normalized_values)

    return torch.cat(out, dim=-2)

class SelfAttention(nn.Module):
    """Implements Dot-Product Self-Attention as used in "Attention is all You Need".
    """
    def __init__(
            self,
            memory_efficient_attention=False,
            q_bucket_size=512,
            k_bucket_size=1024,
        ):
        super(SelfAttention, self).__init__()
        self.use_memory_efficient_attention = memory_efficient_attention
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

    def forward(self, k, q, v):

        if self.use_memory_efficient_attention:
            q, k, v = map(
                lambda t: rearrange(t, 'b n (h d) -> b h n d', h = 12), 
                (q, k, v)
            )

            out = memory_efficient_attention(
                q, k, v, 
                q_bucket_size=self.q_bucket_size, 
                k_bucket_size=self.k_bucket_size
            )
            out = rearrange(out, 'b h n d -> b n (h d)')
            return out
        else:
            d_k = q.size(-1)
            scores = torch.matmul(k, q.transpose(-2, -1)) / math.sqrt(d_k)
            p_attn = F.softmax(scores, dim=-1)
            return torch.matmul(p_attn, v)

class LUAR(PreTrainedModel):
    """Defines the LUAR model.
    """
    config_class = LUARConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.create_transformer()
        self.attn_fn = SelfAttention(
            config.use_memory_efficient_attention,
            config.q_bucket_size,
            config.k_bucket_size,
        )
        self.linear = nn.Linear(self.hidden_size, config.embedding_size)

    def create_transformer(self):
        """Creates the Transformer backbone.
        """
        self.transformer = AutoModel.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v1")
        self.hidden_size = self.transformer.config.hidden_size
        self.num_attention_heads = self.transformer.config.num_attention_heads
        self.dim_head = self.hidden_size // self.num_attention_heads
        
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean Pooling as described in the SBERT paper.
        """
        input_mask_expanded = repeat(attention_mask, 'b l -> b l d', d=self.hidden_size).type(token_embeddings.type())
        sum_embeddings = reduce(token_embeddings * input_mask_expanded, 'b l d -> b d', 'sum')
        sum_mask = torch.clamp(reduce(input_mask_expanded, 'b l d -> b d', 'sum'), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_episode_embeddings(self, input_ids, attention_mask, output_attentions=False, document_batch_size=0):
        """Computes the Author Embedding. 
        """
        B, E, _ = attention_mask.shape

        input_ids = rearrange(input_ids, 'b e l -> (b e) l')
        attention_mask = rearrange(attention_mask, 'b e l -> (b e) l')

        if document_batch_size > 0:
            outputs = {"last_hidden_state": [], "attentions": []}
            for i in range(0, len(input_ids), document_batch_size):
                out = self.transformer(
                    input_ids=input_ids[i:i+document_batch_size],
                    attention_mask=attention_mask[i:i+document_batch_size],
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=output_attentions,
                )
                outputs["last_hidden_state"].append(out["last_hidden_state"])
                if output_attentions:
                    outputs["attentions"].append(out["attentions"])
            outputs["last_hidden_state"] = torch.cat(outputs["last_hidden_state"], dim=0)
            if output_attentions:
                outputs["attentions"] = tuple([torch.cat([x[i] for x in outputs["attentions"]], dim=0) for i in range(len(outputs["attentions"][0]))])
        else:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=False,
                output_attentions=output_attentions,
            )
            
        # at this point, we're embedding individual "comments"
        comment_embeddings = self.mean_pooling(outputs['last_hidden_state'], attention_mask)
        comment_embeddings = rearrange(comment_embeddings, '(b e) l -> b e l', b=B, e=E)

        # aggregate individual comments embeddings into episode embeddings
        episode_embeddings = self.attn_fn(comment_embeddings, comment_embeddings, comment_embeddings)
        episode_embeddings = reduce(episode_embeddings, 'b e l -> b l', 'max')
        
        episode_embeddings = self.linear(episode_embeddings)
        
        if output_attentions:
            return episode_embeddings, outputs["attentions"]

        return episode_embeddings
    
    def forward(self, input_ids, attention_mask, output_attentions=False, document_batch_size=0):
        """Calculates a fixed-length feature vector for a batch of episode samples.
        """
        output = self.get_episode_embeddings(input_ids, attention_mask, output_attentions, document_batch_size)

        return output

class LUARSimilar(LUAR):
    
    def __init__(self, config):
        super().__init__(config)
        self.distance_metric = lambda x, y: 1 - F.cosine_similarity(x, y)
        self.margin = 0.5

    def forward(self, 
                input_ids1, 
                attention_mask1,
                input_ids2, 
                attention_mask2,
                labels):
        """Calculates a fixed-length feature vector for a batch of episode samples.
        """
        # input_ids/attention_mask: [bsz, eps, token]
        output1 = self.get_episode_embeddings(input_ids1, attention_mask1)
        output2 = self.get_episode_embeddings(input_ids2, attention_mask2)
        loss = None
        distances = self.distance_metric(output1, output2)
        logits = 1 - distances
        if labels is not None:
            losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
            loss = losses.mean()

        return LUARSimilarOutput(
            loss=loss,
            logits=logits
        )

# MixFeature Model
class MixFeaturesConfig(PretrainedConfig):
    model_type = "MixFeatures"
    
    def __init__(self,
        embedding_size: int = 1024,
        **kwargs,
    ):
        self.embedding_size = embedding_size
        super().__init__(**kwargs)


@dataclass
class MixFeaturesContrastiveOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    similarity: torch.FloatTensor = None

class MixFeaturesContrastiveModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self,
                 config: PretrainedConfig,
                 authorship_pretrained_path: str,
                 sbert_pretrained_path: str,
                 stel_pretrained_path: str,
                 roberta_pretrained_path):
        super(MixFeaturesContrastiveModel, self).__init__(config)
        self.config = config
        self.authorship_encoder = AutoModel.from_pretrained(authorship_pretrained_path, trust_remote_code=True)
        for p in self.authorship_encoder.parameters():
            p.requires_grad = False
        self.sbert_encoder = AutoModel.from_pretrained(sbert_pretrained_path)
        for p in self.sbert_encoder.parameters():
            p.requires_grad = False
        self.stel_encoder = AutoModel.from_pretrained(stel_pretrained_path)
        for p in self.stel_encoder.parameters():
            p.requires_grad = False
        self.roberta_encoder = AutoModel.from_pretrained(roberta_pretrained_path)
        for p in self.roberta_encoder.parameters():
            p.requires_grad = False
        self.distance_metric = lambda x, y: 1 - F.cosine_similarity(x, y)
    
    def authorship_encode(self,
                          input_ids,
                          attention_mask):
        # input_ids: bsz * utter_num * utter_len
        # attention_mask: bsz * utter_num * utter_len
        # authorship_embeddings bsz * d_model
        bsz, _, _ = input_ids.shape
        authorship_embeddings = self.authorship_encoder(input_ids=input_ids, attention_mask=attention_mask)
        authorship_embeddings = F.normalize(authorship_embeddings, p=2, dim=1).reshape(bsz, -1)
        return authorship_embeddings.detach()
    
    def sbert_encode(self,
                      input_ids,
                      attention_mask,
                      utterance_mask):
        # input_ids: bsz * utter_num * utter_len
        # attention_mask: bsz * utter_num * utter_len
        # utterance_mask: bsz * utter_num
        # sbert_embeddings bsz * d_model
        bsz, utter_num, _ = input_ids.shape
        input_ids=input_ids.reshape(bsz*utter_num, -1).contiguous()
        attention_mask=attention_mask.reshape(bsz*utter_num, -1).contiguous()
        sbert_embeddings = self.sbert_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sbert_embeddings.size()).float()
        sbert_embeddings = torch.sum(sbert_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sbert_embeddings = sbert_embeddings.reshape(bsz, utter_num, -1)
        input_mask_expanded = utterance_mask.unsqueeze(-1).expand(sbert_embeddings.size()).float()
        sbert_embeddings = torch.sum(sbert_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sbert_embeddings = F.normalize(sbert_embeddings, p=2, dim=1).reshape(bsz, -1)
        return sbert_embeddings.detach()

    def stel_encode(self,
                    input_ids,
                    attention_mask,
                    utterance_mask):
        # input_ids: bsz * utter_num * utter_len
        # attention_mask: bsz * utter_num * utter_len
        # utterance_mask: bsz * utter_num
        # stel_embeddings bsz * d_model
        bsz, utter_num, _ = input_ids.shape
        input_ids=input_ids.reshape(bsz*utter_num, -1).contiguous()
        attention_mask=attention_mask.reshape(bsz*utter_num, -1).contiguous()
        stel_embeddings = self.stel_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(stel_embeddings.size()).float()
        stel_embeddings = torch.sum(stel_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        stel_embeddings = stel_embeddings.reshape(bsz, utter_num, -1)
        input_mask_expanded = utterance_mask.unsqueeze(-1).expand(stel_embeddings.size()).float()
        stel_embeddings = torch.sum(stel_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        stel_embeddings = F.normalize(stel_embeddings, p=2, dim=1).reshape(bsz, -1)
        return stel_embeddings.detach()
    
    def roberta_encode(self,
                    input_ids,
                    attention_mask,
                    utterance_mask):
        # input_ids: bsz * utter_num * utter_len
        # attention_mask: bsz * utter_num * utter_len
        # utterance_mask: bsz * utter_num
        # roberta_embeddings bsz * d_model
        bsz, utter_num, _ = input_ids.shape
        input_ids=input_ids.reshape(bsz*utter_num, -1).contiguous()
        attention_mask=attention_mask.reshape(bsz*utter_num, -1).contiguous()
        roberta_embeddings = self.roberta_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(roberta_embeddings.size()).float()
        roberta_embeddings = torch.sum(roberta_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        roberta_embeddings = roberta_embeddings.reshape(bsz, utter_num, -1)
        input_mask_expanded = utterance_mask.unsqueeze(-1).expand(roberta_embeddings.size()).float()
        roberta_embeddings = torch.sum(roberta_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        roberta_embeddings = F.normalize(roberta_embeddings, p=2, dim=1).reshape(bsz, -1)
        return roberta_embeddings.detach()

    def forward(self, 
                authorship_input_ids1,
                authorship_attention_mask1,
                sbert_input_ids1,
                sbert_attention_mask1,
                stel_input_ids1,
                stel_attention_mask1,
                roberta_input_ids1,
                roberta_attention_mask1,
                utter_mask1,
                authorship_input_ids2,
                authorship_attention_mask2,
                sbert_input_ids2,
                sbert_attention_mask2,
                stel_input_ids2,
                stel_attention_mask2,
                roberta_input_ids2,
                roberta_attention_mask2,
                utter_mask2,
                labels=None,
                **kwargs):
        # conv_input_ids/conv_attention_mask: bsz * conv_len * utter_len
        # utters_input_ids/utters_attention_mask: bsz * utter_num * utter_len
        # utter_mask: bsz * utter_num
        # conversation_mask: bsz * conv_len
        # select_utter_mask: bsz * conv_len
        authorship1 = self.authorship_encode(
            input_ids=authorship_input_ids1,
            attention_mask=authorship_attention_mask1,
        ) # bsz * d_model
        authorship2 = self.authorship_encode(
            input_ids=authorship_input_ids2,
            attention_mask=authorship_attention_mask2,
        ) # bsz * d_model
        sbert1 = self.sbert_encode(
            input_ids=sbert_input_ids1,
            attention_mask=sbert_attention_mask1,
            utterance_mask=utter_mask1,
        ) # bsz * d_model
        sbert2 = self.sbert_encode(
            input_ids=sbert_input_ids2,
            attention_mask=sbert_attention_mask2,
            utterance_mask=utter_mask2,
        ) # bsz * d_model
        stel1 = self.stel_encode(
            input_ids=stel_input_ids1,
            attention_mask=stel_attention_mask1,
            utterance_mask=utter_mask1,
        ) # bsz * d_model
        stel2 = self.stel_encode(
            input_ids=stel_input_ids2,
            attention_mask=stel_attention_mask2,
            utterance_mask=utter_mask2,
        ) # bsz * d_model
        roberta1 = self.roberta_encode(
            input_ids=roberta_input_ids1,
            attention_mask=roberta_attention_mask1,
            utterance_mask=utter_mask1,
        ) # bsz * d_model
        roberta2 = self.roberta_encode(
            input_ids=roberta_input_ids2,
            attention_mask=roberta_attention_mask2,
            utterance_mask=utter_mask2,
        ) # bsz * d_model
        speaker1 = torch.cat((sbert1, authorship1, stel1, roberta1), dim=1)
        speaker2 = torch.cat((sbert2, authorship2, stel2, roberta2), dim=1)
        distances = self.distance_metric(speaker1, speaker2)
        similarity = 1 - distances

        return MixFeaturesContrastiveOutput(
            loss=None,
            similarity=similarity
        )

