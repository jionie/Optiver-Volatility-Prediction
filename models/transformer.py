import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder


class TransfomerModel(nn.Module):
    def __init__(self, config):
        super(TransfomerModel, self).__init__()

        self.seq_len = config.hist_window
        cate_col_size = len(config.cate_cols)
        cont_col_size = len(config.cont_cols)

        # if exists category features
        if cate_col_size > 0:
            self.cate_emb = nn.Embedding(config.total_cate_size, config.emb_size, padding_idx=0)

            self.cate_proj = nn.Sequential(
                nn.Linear(config.emb_size * cate_col_size, config.hidden_size // 2),
                nn.LayerNorm(config.hidden_size // 2),
            )

            self.cont_emb = nn.Sequential(
                nn.Linear(cont_col_size, config.hidden_size // 2),
                nn.LayerNorm(config.hidden_size // 2),
            )

        else:

            self.cont_emb = nn.Sequential(
                nn.Linear(cont_col_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
            )

        self.config = BertConfig(
            vocab_size=3,  # not used
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.hidden_size,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
        )

        self.encoder = BertEncoder(self.config)

        def get_reg():
            return nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.Dropout(config.dropout),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.Dropout(config.dropout),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.target_size),
            )

        self.reg_layer = get_reg()

    def forward(self, cate_x, cont_x, mask):

        batch_size = cont_x.size(0)

        cont_emb = self.cont_emb(cont_x)

        if cate_x is not None:
            cate_emb = self.cate_emb(cate_x).view(batch_size, self.seq_len, -1)
            cate_emb = self.cate_proj(cate_emb)

            seq_emb = torch.cat([cate_emb, cont_emb], 2)
        else:
            seq_emb = cont_emb

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers

        encoded_layers = self.encoder(seq_emb, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]
        sequence_output = sequence_output[:, -1]

        pred_y = self.reg_layer(sequence_output)

        return pred_y


class LSTMATTNModel(nn.Module):
    def __init__(self, config):
        super(LSTMATTNModel, self).__init__()

        self.seq_len = config.hist_window

        cate_col_size = len(config.cate_cols)
        cont_col_size = len(config.cont_cols)
        self.cate_emb = nn.Embedding(config.total_cate_size, config.emb_size, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(config.emb_size * cate_col_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
        )
        self.cont_emb = nn.Sequential(
            nn.Linear(cont_col_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
        )

        self.encoder = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            1,
            dropout=config.dropout,
            batch_first=True
        )

        self.config = BertConfig(
            vocab_size=3,  # not used
            hidden_size=config.hidden_size,
            num_hidden_layers=1,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.hidden_size,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
        )
        self.attn = BertEncoder(self.config)

        def get_reg():
            return nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.Dropout(config.dropout),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.Dropout(config.dropout),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.target_size),
            )

        self.reg_layer = get_reg()

    def forward(self, cate_x, cont_x, mask):

        batch_size = cont_x.size(0)

        cont_emb = self.cont_emb(cont_x)

        if cate_x is not None:
            cate_emb = self.cate_emb(cate_x).view(batch_size, self.seq_len, -1)
            cate_emb = self.cate_proj(cate_emb)

            seq_emb = torch.cat([cate_emb, cont_emb], 2)
        else:
            seq_emb = cont_emb

        output, _ = self.encoder(seq_emb)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers

        encoded_layers = self.attn(output, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]
        sequence_output = sequence_output[:, -1]
        pred_y = self.reg_layer(sequence_output)

        return pred_y
