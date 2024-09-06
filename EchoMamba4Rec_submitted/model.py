import torch
from torch import nn
from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

class EchoMamba4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(EchoMamba4Rec, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        
        # Hyperparameters for Mamba block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
            
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.mamba_layers = nn.ModuleList([
            BiMambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
                max_seq_length=self.max_seq_length
            ) for _ in range(self.num_layers)
        ])
        
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)
        
        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)
        
        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
    
class BiMambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers, max_seq_length):
        super().__init__()
        self.num_layers = num_layers
        
        
        self.filter_layer = FilterLayer(max_seq_length=max_seq_length, hidden_size=d_model, dropout_prob=dropout)

        self.norms_forward = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norms_backward = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        self.mamba_forwards = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)
        ])
        self.mamba_backwards = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        self.glu = GLU(d_model=d_model, dropout=dropout)
        self.multi_query_transformer_block = MultiQueryTransformerBlock(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )

    def forward(self, input_tensor):
        
        x=input_tensor
        x = self.filter_layer(x)
        
        for i in range(self.num_layers):
            forward_states = self.mamba_forwards[i](x)
            forward_states = self.norms_forward[i](self.dropout(forward_states) + x)

            reversed_input = torch.flip(x, [1])  
            backward_states = self.mamba_backwards[i](reversed_input)
            backward_states = torch.flip(backward_states, [1]) 
            backward_states = self.norms_backward[i](self.dropout(backward_states) + x)

            x = forward_states + backward_states

        x = self.glu(x)
       
        
        return x


class FilterLayer(nn.Module):
    def __init__(self, max_seq_length, hidden_size, dropout_prob):
        super(FilterLayer, self).__init__()
        
        self.complex_weight = nn.Parameter(
            torch.randn(1, max_seq_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02
        )
        self.out_dropout = nn.Dropout(dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class MultiQueryTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(MultiQueryTransformerBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.glu = GLU(d_model, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Query Attention
        x_transposed = x.permute(1, 0, 2)  # Change shape from [B, S, D] to [S, B, D]
        attn_output, _ = self.multihead_attn(x_transposed, x_transposed, x_transposed)
        attn_output = attn_output.permute(1, 0, 2)  # Change shape back to [B, S, D]
        x = x + attn_output
        x = self.norm1(x)
        x = self.dropout1(x)

        # Feed Forward
        glu_output = self.glu(x)
        x = x + glu_output
        x = self.norm2(x)
        x = self.dropout2(x)

        return x
class GLU(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super(GLU, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model * 2)  
        self.fc2 = nn.Linear(d_model, d_model)  
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x):
        x_transformed = self.fc1(x)
        value, gate = x_transformed.chunk(2, dim=-1)  
        gated_value = value * torch.sigmoid(gate)
        gated_value = self.fc2(gated_value)  
        return self.LayerNorm(self.dropout(gated_value + x))