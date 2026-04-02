import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from Layers import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphCodeBERT(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(GraphCodeBERT, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.w_embeddings = self.encoder.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.graphEmb = GraphEmbedding(feature_dim_size=768, hidden_size=256, dropout=config.hidden_dropout_prob)
        self.query = 0
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = PredictionClassification(config, args, input_size= 1024)
        
        # 新增：冻结 RoBERTa 前 n 层（例如冻结前 6 层）
        n_freeze_layers = 8  # ← 你可根据实验调整：0=全放开，12=全冻结
        if n_freeze_layers > 0:
            # 冻结 embeddings（可选，通常保留）
            #for param in self.encoder.embeddings.parameters():
                #param.requires_grad = False

            # 冻结 Transformer 编码器的前 n 层
            for layer in self.encoder.encoder.layer[:n_freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        

    def forward(self, inputs_ids=None, attn_mask=None, position_idx=None, labels=None, ast_adj=None, cfg_adj=None, pdg_adj=None, node_features=None, node_mask=None):
        g_emb = self.graphEmb(node_features.to(device).float(), ast_adj.to(device).float(), cfg_adj.to(device).float(), pdg_adj.to(device).float(), node_mask.to(device).float())
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)

        inputs_embeddings = self.encoder.embeddings.word_embeddings(inputs_ids)

        vec = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask, position_ids=position_idx)[0][:, 0, :]
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