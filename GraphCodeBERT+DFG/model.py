import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from Layers import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CodeT5GraphModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5GraphModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.graphEmb = GraphEmbedding(feature_dim_size=768, hidden_size=256, dropout=config.dropout_rate)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = PredictionClassification(config, args, input_size= 1024)
        
        # 新增：冻结 CodeT5 encoder 前 n 层（例如冻结前 6 层）
        n_freeze_layers = 8  # ← 你可根据实验调整：0=全放开，12=全冻结
        if n_freeze_layers > 0:
            for block in self.encoder.encoder.block[:n_freeze_layers]:
                for param in block.parameters():
                    param.requires_grad = False

    def get_t5_vec(self, source_ids):
        """
        走完整 encoder-decoder，取 decoder 最后一层 <eos> 位置的向量。
        labels=source_ids 触发 teacher forcing，让 decoder 有输出的 hidden states。
        """
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(
            input_ids=source_ids,
            attention_mask=attention_mask,
            labels=source_ids,
            decoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        # decoder 最后一层的所有 token 的隐状态，shape: [B, seq_len, hidden]
        hidden_states = outputs.decoder_hidden_states[-1]

        # 找每个样本中 <eos> token 的位置
        eos_mask = source_ids.eq(self.tokenizer.eos_token_id)

        # 校验：每个样本必须有且只有相同数量的 <eos>（通常是1个）
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        # 取每个样本最后一个 <eos> 的向量，shape: [B, hidden]
        vec = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1)
        )[:, -1, :]

        return vec

    def forward(self, inputs_ids=None, labels=None, ast_adj=None, cfg_adj=None, pdg_adj=None, node_features=None, node_mask=None):
        g_emb = self.graphEmb(node_features.to(device).float(), ast_adj.to(device).float(), cfg_adj.to(device).float(), pdg_adj.to(device).float(), node_mask.to(device).float())

        vec = self.get_t5_vec(inputs_ids)
        outputs = self.classifier(torch.cat((vec, g_emb), dim=1))
        logits = outputs.squeeze(-1)

        if labels is not None:
            labels = labels.float()
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


def distill_loss(logits, knowledge, temperature=10.0):
    loss = F.kl_div(F.log_softmax(logits/temperature), F.softmax(knowledge /
                    temperature), reduction="batchmean") * (temperature**2)
    # Equivalent to cross_entropy for soft labels, from https://github.com/huggingface/transformers/blob/50792dbdcccd64f61483ec535ff23ee2e4f9e18d/examples/distillation/distiller.py#L330

    return loss