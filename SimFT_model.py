import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.configuration_bert import BertConfig
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel


class SimFT_ffn(BertPreTrainedModel):
    
    def __init__(self, config, is_initial=False, bert_path=None):
        super().__init__(config)
        self.config = config

        self.linear = nn.Linear(config.hidden_size * 2, 1)
        self.ffn_sem = nn.Linear(config.hidden_size, config.hidden_size)
        self.ffn_syn = nn.Linear(config.hidden_size, config.hidden_size)
        # Initialize weights and apply final processing
        self.post_init()

        if is_initial:
            self.encoder_sem = BertModel(config, add_pooling_layer=False)
            self.encoder_sem =  self.encoder_sem.from_pretrained(bert_path)
            self.encoder_syn = BertModel(config, add_pooling_layer=False)
            self.encoder_syn =  self.encoder_syn.from_pretrained(bert_path)
        else:
            self.encoder_sem = BertModel(config, add_pooling_layer=False)
            self.encoder_syn = BertModel(config, add_pooling_layer=False)


    def forward(
        self,
        sem_input_ids: Optional[torch.Tensor] = None,
        sem_attention_mask: Optional[torch.Tensor] = None,
        sem_token_type_ids: Optional[torch.Tensor] = None,
        sem_position_ids: Optional[torch.Tensor] = None,
        syn_input_ids: Optional[torch.Tensor] = None,
        syn_attention_mask: Optional[torch.Tensor] = None,
        syn_token_type_ids: Optional[torch.Tensor] = None,
        syn_position_ids: Optional[torch.Tensor] = None,
    ):

        sem_encoder_output = self.encoder_sem(
            input_ids=sem_input_ids,
            attention_mask=sem_attention_mask,
            position_ids=sem_position_ids,
            token_type_ids=sem_token_type_ids,
            return_dict=True
        )["last_hidden_state"]
        syn_encoder_output = self.encoder_syn(
            input_ids=syn_input_ids,
            attention_mask=syn_attention_mask,
            position_ids=syn_position_ids,
            token_type_ids=syn_token_type_ids,
            return_dict=True
        )["last_hidden_state"]

        # print(f"@@@1 sem_encoder_output-size {sem_encoder_output.size()}, syn_encoder_output-size {syn_encoder_output.size()} {sem_attention_mask.size()}")

        sem_ffn = self.ffn_sem(sem_encoder_output)
        syn_ffn = self.ffn_syn(syn_encoder_output)
        # print(f"@@@2 sem_ffn: {sem_ffn.size()}, syn_ffn: {syn_ffn.size()}")

        sim_matrix = torch.matmul(sem_ffn, syn_ffn.transpose(-1, -2))  # bsz*sen-len*syn-len
        # print(f"@@@3 sim_matrix: {sim_matrix.size()}")

        sem_sim_mask = (1.0 - sem_attention_mask.int()) * torch.finfo(sim_matrix.dtype).min
        syn_sim_mask = (1.0 - syn_attention_mask.int()) * torch.finfo(sim_matrix.dtype).min
        # print(f"@@@4 sem_sim_mask: {sem_sim_mask.size()} syn_sim_mask: {syn_sim_mask.size()}")
        # print(sem_sim_mask)

        sem_syn_sim_score = sim_matrix + syn_sim_mask.unsqueeze(1)
        syn_sem_sim_score = sim_matrix.transpose(-1, -2) + sem_sim_mask.unsqueeze(1)
        # print(f"@@@5 sem_syn_sim_score.size(): {sem_syn_sim_score.size(), syn_sem_sim_score.size()}")

        sem_embed = ((sem_ffn * torch.max(sem_syn_sim_score, dim=-1, keepdim=True).values) * sem_attention_mask.unsqueeze(-1)).sum(1) / sem_attention_mask.unsqueeze(-1).sum(1)
        syn_embed = ((syn_ffn * torch.max(syn_sem_sim_score, dim=-1, keepdim=True).values) * syn_attention_mask.unsqueeze(-1)).sum(1) / syn_attention_mask.unsqueeze(-1).sum(1)
        # print(f"@@@6 sem_embed： {sem_embed.size()}, syn_embed: {syn_embed.size()} {torch.cat((sem_embed, syn_embed), dim=-1).size()}")

        output = self.linear(torch.cat((sem_embed, syn_embed), dim=-1))

        return {
            "output": output
        }



class SimFT_ffn_Roberta(RobertaPreTrainedModel):
    
    def __init__(self, config, is_initial=False, bert_path=None):
        super().__init__(config)
        self.config = config

        self.linear = nn.Linear(config.hidden_size * 2, 1)
        self.ffn_sem = nn.Linear(config.hidden_size, config.hidden_size)
        self.ffn_syn = nn.Linear(config.hidden_size, config.hidden_size)
        # Initialize weights and apply final processing
        self.post_init()

        if is_initial:
            self.encoder_sem = RobertaModel(config, add_pooling_layer=False)
            self.encoder_sem =  self.encoder_sem.from_pretrained(bert_path)
            self.encoder_syn = RobertaModel(config, add_pooling_layer=False)
            self.encoder_syn =  self.encoder_syn.from_pretrained(bert_path)
        else:
            self.encoder_sem = RobertaModel(config, add_pooling_layer=False)
            self.encoder_syn = RobertaModel(config, add_pooling_layer=False)


    def forward(
        self,
        sem_input_ids: Optional[torch.Tensor] = None,
        sem_attention_mask: Optional[torch.Tensor] = None,
        sem_token_type_ids: Optional[torch.Tensor] = None,
        sem_position_ids: Optional[torch.Tensor] = None,
        syn_input_ids: Optional[torch.Tensor] = None,
        syn_attention_mask: Optional[torch.Tensor] = None,
        syn_token_type_ids: Optional[torch.Tensor] = None,
        syn_position_ids: Optional[torch.Tensor] = None,
    ):

        sem_encoder_output = self.encoder_sem(
            input_ids=sem_input_ids,
            attention_mask=sem_attention_mask,
            position_ids=sem_position_ids,
            token_type_ids=sem_token_type_ids,
            return_dict=True
        )["last_hidden_state"]
        syn_encoder_output = self.encoder_syn(
            input_ids=syn_input_ids,
            attention_mask=syn_attention_mask,
            position_ids=syn_position_ids,
            token_type_ids=syn_token_type_ids,
            return_dict=True
        )["last_hidden_state"]

        # print(f"@@@1 sem_encoder_output-size {sem_encoder_output.size()}, syn_encoder_output-size {syn_encoder_output.size()} {sem_attention_mask.size()}")

        sem_ffn = self.ffn_sem(sem_encoder_output)
        syn_ffn = self.ffn_syn(syn_encoder_output)
        # print(f"@@@2 sem_ffn: {sem_ffn.size()}, syn_ffn: {syn_ffn.size()}")

        sim_matrix = torch.matmul(sem_ffn, syn_ffn.transpose(-1, -2))  # bsz*sen-len*syn-len
        # print(f"@@@3 sim_matrix: {sim_matrix.size()}")

        sem_sim_mask = (1.0 - sem_attention_mask.int()) * torch.finfo(sim_matrix.dtype).min
        syn_sim_mask = (1.0 - syn_attention_mask.int()) * torch.finfo(sim_matrix.dtype).min
        # print(f"@@@4 sem_sim_mask: {sem_sim_mask.size()} syn_sim_mask: {syn_sim_mask.size()}")
        # print(sem_sim_mask)

        sem_syn_sim_score = sim_matrix + syn_sim_mask.unsqueeze(1)
        syn_sem_sim_score = sim_matrix.transpose(-1, -2) + sem_sim_mask.unsqueeze(1)
        # print(f"@@@5 sem_syn_sim_score.size(): {sem_syn_sim_score.size(), syn_sem_sim_score.size()}")

        sem_embed = ((sem_ffn * torch.max(sem_syn_sim_score, dim=-1, keepdim=True).values) * sem_attention_mask.unsqueeze(-1)).sum(1) / sem_attention_mask.unsqueeze(-1).sum(1)
        syn_embed = ((syn_ffn * torch.max(syn_sem_sim_score, dim=-1, keepdim=True).values) * syn_attention_mask.unsqueeze(-1)).sum(1) / syn_attention_mask.unsqueeze(-1).sum(1)
        # print(f"@@@6 sem_embed： {sem_embed.size()}, syn_embed: {syn_embed.size()} {torch.cat((sem_embed, syn_embed), dim=-1).size()}")

        output = self.linear(torch.cat((sem_embed, syn_embed), dim=-1))

        return {
            "output": output
        }



class SimFT(BertPreTrainedModel):
    
    def __init__(self, config, is_initial=False, bert_path=None):
        super().__init__(config)
        self.config = config

        self.linear = nn.Linear(config.hidden_size * 2, 1)
        # Initialize weights and apply final processing
        self.post_init()

        if is_initial:
            self.encoder_sem = BertModel(config, add_pooling_layer=False)
            self.encoder_sem =  self.encoder_sem.from_pretrained(bert_path)
            self.encoder_syn = BertModel(config, add_pooling_layer=False)
            self.encoder_syn =  self.encoder_syn.from_pretrained(bert_path)
        else:
            self.encoder_sem = BertModel(config, add_pooling_layer=False)
            self.encoder_syn = BertModel(config, add_pooling_layer=False)


    def forward(
        self,
        sem_input_ids: Optional[torch.Tensor] = None,
        sem_attention_mask: Optional[torch.Tensor] = None,
        sem_token_type_ids: Optional[torch.Tensor] = None,
        sem_position_ids: Optional[torch.Tensor] = None,
        syn_input_ids: Optional[torch.Tensor] = None,
        syn_attention_mask: Optional[torch.Tensor] = None,
        syn_token_type_ids: Optional[torch.Tensor] = None,
        syn_position_ids: Optional[torch.Tensor] = None,
    ):

        sem_encoder_output = self.encoder_sem(
            input_ids=sem_input_ids,
            attention_mask=sem_attention_mask,
            position_ids=sem_position_ids,
            token_type_ids=sem_token_type_ids,
            return_dict=True
        )["last_hidden_state"]
        syn_encoder_output = self.encoder_syn(
            input_ids=syn_input_ids,
            attention_mask=syn_attention_mask,
            position_ids=syn_position_ids,
            token_type_ids=syn_token_type_ids,
            return_dict=True
        )["last_hidden_state"]

        # print(f"@@@1 sem_encoder_output-size {sem_encoder_output.size()}, syn_encoder_output-size {syn_encoder_output.size()} {sem_attention_mask.size()}")


        sim_matrix = torch.matmul(sem_encoder_output, syn_encoder_output.transpose(-1, -2))  # bsz*sen-len*syn-len
        # print(f"@@@3 sim_matrix: {sim_matrix.size()}")

        sem_sim_mask = (1.0 - sem_attention_mask.int()) * torch.finfo(sim_matrix.dtype).min
        syn_sim_mask = (1.0 - syn_attention_mask.int()) * torch.finfo(sim_matrix.dtype).min
        # print(f"@@@4 sem_sim_mask: {sem_sim_mask.size()} syn_sim_mask: {syn_sim_mask.size()}")
        # print(sem_sim_mask)

        sem_syn_sim_score = sim_matrix + syn_sim_mask.unsqueeze(1)
        syn_sem_sim_score = sim_matrix.transpose(-1, -2) + sem_sim_mask.unsqueeze(1)
        # print(f"@@@5 sem_syn_sim_score.size(): {sem_syn_sim_score.size(), syn_sem_sim_score.size()}")

        sem_embed = ((sem_encoder_output * torch.max(sem_syn_sim_score, dim=-1, keepdim=True).values) * sem_attention_mask.unsqueeze(-1)).sum(1) / sem_attention_mask.unsqueeze(-1).sum(1)
        syn_embed = ((syn_encoder_output * torch.max(syn_sem_sim_score, dim=-1, keepdim=True).values) * syn_attention_mask.unsqueeze(-1)).sum(1) / syn_attention_mask.unsqueeze(-1).sum(1)
        # print(f"@@@6 sem_embed： {sem_embed.size()}, syn_embed: {syn_embed.size()} {torch.cat((sem_embed, syn_embed), dim=-1).size()}")

        output = self.linear(torch.cat((sem_embed, syn_embed), dim=-1))

        return {
            "output": output
        }


class SimFT_ffn_softmax(BertPreTrainedModel):
    
    def __init__(self, config, is_initial=False, bert_path=None):
        super().__init__(config)
        self.config = config

        self.linear = nn.Linear(config.hidden_size * 2, 1)
        self.ffn_sem = nn.Linear(config.hidden_size, config.hidden_size)
        self.ffn_syn = nn.Linear(config.hidden_size, config.hidden_size)
        # Initialize weights and apply final processing
        self.post_init()

        if is_initial:
            self.encoder_sem = BertModel(config, add_pooling_layer=False)
            self.encoder_sem =  self.encoder_sem.from_pretrained(bert_path)
            self.encoder_syn = BertModel(config, add_pooling_layer=False)
            self.encoder_syn =  self.encoder_syn.from_pretrained(bert_path)
        else:
            self.encoder_sem = BertModel(config, add_pooling_layer=False)
            self.encoder_syn = BertModel(config, add_pooling_layer=False)


    def forward(
        self,
        sem_input_ids: Optional[torch.Tensor] = None,
        sem_attention_mask: Optional[torch.Tensor] = None,
        sem_token_type_ids: Optional[torch.Tensor] = None,
        sem_position_ids: Optional[torch.Tensor] = None,
        syn_input_ids: Optional[torch.Tensor] = None,
        syn_attention_mask: Optional[torch.Tensor] = None,
        syn_token_type_ids: Optional[torch.Tensor] = None,
        syn_position_ids: Optional[torch.Tensor] = None,
    ):

        sem_encoder_output = self.encoder_sem(
            input_ids=sem_input_ids,
            attention_mask=sem_attention_mask,
            position_ids=sem_position_ids,
            token_type_ids=sem_token_type_ids,
            return_dict=True
        )["last_hidden_state"]
        syn_encoder_output = self.encoder_syn(
            input_ids=syn_input_ids,
            attention_mask=syn_attention_mask,
            position_ids=syn_position_ids,
            token_type_ids=syn_token_type_ids,
            return_dict=True
        )["last_hidden_state"]

        # print(f"@@@1 sem_encoder_output-size {sem_encoder_output.size()}, syn_encoder_output-size {syn_encoder_output.size()} {sem_attention_mask.size()}")

        sem_ffn = self.ffn_sem(sem_encoder_output)
        syn_ffn = self.ffn_syn(syn_encoder_output)
        # print(f"@@@2 sem_ffn: {sem_ffn.size()}, syn_ffn: {syn_ffn.size()}")

        sim_matrix = torch.matmul(sem_ffn, syn_ffn.transpose(-1, -2))  # bsz*sen-len*syn-len
        # print(f"@@@3 sim_matrix: {sim_matrix.size()}")

        sem_sim_mask = (1.0 - sem_attention_mask.int()) * torch.finfo(sim_matrix.dtype).min   # bsz*sen-len
        syn_sim_mask = (1.0 - syn_attention_mask.int()) * torch.finfo(sim_matrix.dtype).min   # bsz*syn-len
        # print(f"@@@4 sem_sim_mask: {sem_sim_mask.size()} syn_sim_mask: {syn_sim_mask.size()}")
        # print(sem_sim_mask)

        sem_syn_sim_score = sim_matrix + syn_sim_mask.unsqueeze(1)
        syn_sem_sim_score = sim_matrix.transpose(-1, -2) + sem_sim_mask.unsqueeze(1)
        # print(f"@@@5 sem_syn_sim_score.size(): {sem_syn_sim_score.size(), syn_sem_sim_score.size()}")

        softmax = nn.Softmax(dim=-1)
        sem_score = softmax(torch.max(sem_syn_sim_score, dim=-1, keepdim=False).values + sem_sim_mask)
        syn_score = softmax(torch.max(syn_sem_sim_score, dim=-1, keepdim=False).values + syn_sim_mask)
        # print(f"@@@ sem_score.size(): {sem_score.size()}, syn_score.size(): {syn_score.size()}")

        sem_embed = ((sem_ffn * sem_score.unsqueeze(-1)) * sem_attention_mask.unsqueeze(-1)).sum(1) / sem_attention_mask.unsqueeze(-1).sum(1)
        syn_embed = ((syn_ffn * syn_score.unsqueeze(-1)) * syn_attention_mask.unsqueeze(-1)).sum(1) / syn_attention_mask.unsqueeze(-1).sum(1)
        # print(f"@@@6 sem_embed： {sem_embed.size()}, syn_embed: {syn_embed.size()} {torch.cat((sem_embed, syn_embed), dim=-1).size()}")

        output = self.linear(torch.cat((sem_embed, syn_embed), dim=-1))

        return {
            "output": output
        }


class SimFT_softmax(BertPreTrainedModel):
    
    def __init__(self, config, is_initial=False, bert_path=None):
        super().__init__(config)
        self.config = config

        self.linear = nn.Linear(config.hidden_size * 2, 1)
        # Initialize weights and apply final processing
        self.post_init()

        if is_initial:
            self.encoder_sem = BertModel(config, add_pooling_layer=False)
            self.encoder_sem =  self.encoder_sem.from_pretrained(bert_path)
            self.encoder_syn = BertModel(config, add_pooling_layer=False)
            self.encoder_syn =  self.encoder_syn.from_pretrained(bert_path)
        else:
            self.encoder_sem = BertModel(config, add_pooling_layer=False)
            self.encoder_syn = BertModel(config, add_pooling_layer=False)


    def forward(
        self,
        sem_input_ids: Optional[torch.Tensor] = None,
        sem_attention_mask: Optional[torch.Tensor] = None,
        sem_token_type_ids: Optional[torch.Tensor] = None,
        sem_position_ids: Optional[torch.Tensor] = None,
        syn_input_ids: Optional[torch.Tensor] = None,
        syn_attention_mask: Optional[torch.Tensor] = None,
        syn_token_type_ids: Optional[torch.Tensor] = None,
        syn_position_ids: Optional[torch.Tensor] = None,
    ):

        sem_encoder_output = self.encoder_sem(
            input_ids=sem_input_ids,
            attention_mask=sem_attention_mask,
            position_ids=sem_position_ids,
            token_type_ids=sem_token_type_ids,
            return_dict=True
        )["last_hidden_state"]
        syn_encoder_output = self.encoder_syn(
            input_ids=syn_input_ids,
            attention_mask=syn_attention_mask,
            position_ids=syn_position_ids,
            token_type_ids=syn_token_type_ids,
            return_dict=True
        )["last_hidden_state"]

        # print(f"@@@1 sem_encoder_output-size {sem_encoder_output.size()}, syn_encoder_output-size {syn_encoder_output.size()} {sem_attention_mask.size()}")

        sim_matrix = torch.matmul(sem_encoder_output, syn_encoder_output.transpose(-1, -2))  # bsz*sen-len*syn-len
        # print(f"@@@3 sim_matrix: {sim_matrix.size()}")

        sem_sim_mask = (1.0 - sem_attention_mask.int()) * torch.finfo(sim_matrix.dtype).min   # bsz*sen-len
        syn_sim_mask = (1.0 - syn_attention_mask.int()) * torch.finfo(sim_matrix.dtype).min   # bsz*syn-len
        # print(f"@@@4 sem_sim_mask: {sem_sim_mask.size()} syn_sim_mask: {syn_sim_mask.size()}")
        # print(sem_sim_mask)

        sem_syn_sim_score = sim_matrix + syn_sim_mask.unsqueeze(1)
        syn_sem_sim_score = sim_matrix.transpose(-1, -2) + sem_sim_mask.unsqueeze(1)
        # print(f"@@@5 sem_syn_sim_score.size(): {sem_syn_sim_score.size(), syn_sem_sim_score.size()}")
        
        softmax = nn.Softmax(dim=-1)
        sem_score = softmax(torch.max(sem_syn_sim_score, dim=-1, keepdim=False).values + sem_sim_mask)
        syn_score = softmax(torch.max(syn_sem_sim_score, dim=-1, keepdim=False).values + syn_sim_mask)
        # print(f"@@@ sem_score.size(): {sem_score.size()}, syn_score.size(): {syn_score.size()}")

        sem_embed = ((sem_encoder_output * sem_score.unsqueeze(-1)) * sem_attention_mask.unsqueeze(-1)).sum(1) / sem_attention_mask.unsqueeze(-1).sum(1)
        syn_embed = ((syn_encoder_output * syn_score.unsqueeze(-1)) * syn_attention_mask.unsqueeze(-1)).sum(1) / syn_attention_mask.unsqueeze(-1).sum(1)
        # print(f"@@@6 sem_embed： {sem_embed.size()}, syn_embed: {syn_embed.size()} {torch.cat((sem_embed, syn_embed), dim=-1).size()}")

        output = self.linear(torch.cat((sem_embed, syn_embed), dim=-1))

        return {
            "output": output
        }

