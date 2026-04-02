from __future__ import absolute_import, division, print_function

import os
import sys
import json
import torch
import random
import logging
import argparse
import numpy as np
import multiprocessing
sys.path.append('./parser')


import pickle
from cpg_nx_preprocess import preprocess_cpg_sub
import torch.nn.functional as F

from tqdm import tqdm
from model import GraphCodeBERT
from run_parser import extract_dataflow
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,RobertaModel,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

cpu_cont = multiprocessing.cpu_count()
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 position_idx,
                 idx,
                 label,
                 cpg_object

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.idx = str(idx)
        self.label = label
        self.cpg_object = cpg_object


def convert_examples_to_features_graphcodebert(js, tokenizer, args, pkl_dir_path=None):
    # source
    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length + args.data_flow_length - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]


    padding_length = args.code_length + args.data_flow_length - len(source_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length



    # ---------------------- 新增部分：加载.pkl文件，获取cpg_object（G） ----------------------
    cpg_object = None  # 初始化cpg_object为None
    if pkl_dir_path is not None:  # 如果传入了.pkl目录路径，才加载文件
        # 1. 拼接当前js对应的.pkl文件路径（根据之前生成.pkl的命名规则拼接）
        # 之前生成.pkl的命名格式：{export_format}-{idx}-{target}.pkl
        export_format = "cpg14"
        idx = js['idx']
        target = js['target']
        pkl_file_name = f"{export_format}-{idx}-{target}.pkl"
        pkl_file_path = os.path.join(pkl_dir_path, pkl_file_name)

        # 2. 加载.pkl文件，反序列化得到NetworkX对象G
        try:
            with open(pkl_file_path, 'rb') as f:
                cpg_object = pickle.load(f)  # cpg_object就是之前的G
        except FileNotFoundError:
            print(f"警告：未找到对应的.pkl文件，cpg_object保持None：{pkl_file_path}")
        except Exception as e:
            print(f"警告：加载.pkl文件失败，cpg_object保持None，异常信息：{e}")
    # -------------------------------------------------------------------------------------

    return InputFeatures(source_tokens, source_ids, position_idx, js['idx'], js['target'], cpg_object=cpg_object)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, encoder_model, file_path=None):
        self.examples = []
        self.args = args
        self.tokenizer = tokenizer
        self.cpg_embeddings = encoder_model.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()

        with open(file_path) as f:
            # 1. 读取所有行并缓存
            all_lines = [line.strip() for line in f if line.strip()]  # 过滤空行
            # 2. 试跑截断：仅保留前100条（区分训练/验证集）
            if 'train' in file_path.lower():  # lower()兼容大小写（如Train/train）
                split_name = 'train'
            elif 'valid' in file_path.lower():
                split_name = 'valid'
            elif 'test' in file_path.lower():
                split_name = 'test'
            # 3. 遍历截断后的行处理样本
            for line in tqdm(all_lines, desc=f"Loading {file_path}"):
                js = json.loads(line)
                current_script_patho = os.path.abspath(__file__)
                current_script_diro = os.path.dirname(current_script_patho)
                project_root_diro = os.path.dirname(current_script_diro)
                pkl_dir_path = os.path.join(project_root_diro, 'joerntool', 'joern', 'data', 'runtrydata',
                                            f'cpg14_output_pickle_{split_name}')
                self.examples.append(convert_examples_to_features_graphcodebert(js, self.tokenizer, args, pkl_dir_path))

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                logger.info("cpg_object: {}".format(list(example.cpg_object.nodes)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=np.bool_)
        # calculate begin index of node and max length of input


        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:max_length, :max_length] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True


        ast_adj, cfg_adj, pdg_adj, cpg_node_features = preprocess_cpg_sub(self.examples[item],
                                                                          self.tokenizer, self.cpg_embeddings)

        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(self.examples[item].label),
                torch.tensor(ast_adj),
                torch.tensor(cfg_adj),
                torch.tensor(pdg_adj),
                torch.tensor(cpg_node_features))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 加一个批加载器collate_fn
def collate_fn(batch):
    # 解包
    input_ids_list, attn_mask_list, pos_idx_list = [], [], []
    labels_list = []
    ast_adj_list, cfg_adj_list, pdg_adj_list = [], [], []
    feats_list = []

    # 记录每个样本的真实节点数
    num_nodes_list = []

    for item in batch:
        input_ids_list.append(item[0])
        attn_mask_list.append(item[1])
        pos_idx_list.append(item[2])
        labels_list.append(item[3])
        ast_adj_list.append(item[4])
        cfg_adj_list.append(item[5])
        pdg_adj_list.append(item[6])
        feats_list.append(item[7])
        num_nodes_list.append(item[7].size(0))  # N

    # ========== 文本部分 ==========
    # 如果你已经在 tokenizer 中 pad 到固定长度：
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attn_mask_list)
    position_idx = torch.stack(pos_idx_list)
    labels = torch.stack(labels_list)

    # ========== 图部分 ==========
    max_nodes = max(f.size(0) for f in feats_list)

    def pad_to_max(tensor_2d):
        N = tensor_2d.size(0)
        pad_size = max_nodes - N
        padded = F.pad(tensor_2d, (0, pad_size, 0, pad_size), value=0)
        # return padded.unsqueeze(-1)  # [M, M] -> [M, M, 1]
        return padded  # [M, M]

    ast_batch = torch.stack([pad_to_max(adj) for adj in ast_adj_list])
    cfg_batch = torch.stack([pad_to_max(adj) for adj in cfg_adj_list])
    pdg_batch = torch.stack([pad_to_max(adj) for adj in pdg_adj_list])

    feats_batch = torch.stack([
        F.pad(feat, (0, 0, 0, max_nodes - feat.size(0)), value=0)
        for feat in feats_list
    ])

    # node_mask → [B, M]， 注意掩码时需要广播一下，通过unsqueeze(-1)弄到合适的维度
    node_mask = torch.zeros(len(batch), max_nodes, dtype=torch.bool)
    for i, N in enumerate(num_nodes_list):
        node_mask[i, :N] = 1

    # 注意train中取数据时因为从元组变字典了，取数据的方式要变！！！！！！！！！
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_idx': position_idx,
        'labels': labels,
        'ast_adj': ast_batch,
        'cfg_adj': cfg_batch,
        'pdg_adj': pdg_batch,
        'node_features': feats_batch,
        'node_mask': node_mask
    }


def train(args, train_dataset, model, tokenizer, modelencoder):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=0, pin_memory=True,
                                  collate_fn=collate_fn)
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    
    
    # === FROZEN LAYERS VERIFICATION ===
    print("\n VERIFICATION: Frozen parameters in GraphCodeBERT")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_percent = (total_params - trainable_params) / total_params * 100
    print(f"   Total params: {total_params:,} | Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%) | Frozen: {frozen_percent:.1f}%")

    # Print encoder layer status
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'encoder') and hasattr(model.encoder.encoder, 'layer'):
        for i, layer in enumerate(model.encoder.encoder.layer):
            layer_trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            status = "FROZEN" if layer_trainable == 0 else "TRAINABLE"
            print(f"   Encoder Layer {i}: {status}")
    print("===================================\n")
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            # 从 batch dict 中提取并移到设备上(原来是元组，现在是字典)
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            position_idx = batch['position_idx'].to(args.device)
            labels = batch['labels'].to(args.device)

            ast_adj = batch['ast_adj'].to(args.device)  # [B, M, M]
            cfg_adj = batch['cfg_adj'].to(args.device)  # [B, M, M]
            pdg_adj = batch['pdg_adj'].to(args.device)  # [B, M, M]
            node_features = batch['node_features'].to(args.device)  # [B, M, F]
            node_mask = batch['node_mask'].to(args.device)  # [B, M]

            model.train()

            # 传参给模型的时候也要改
            loss, logits = model(input_ids, attention_mask, position_idx, labels, ast_adj, cfg_adj, pdg_adj,
                                 node_features, node_mask)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()

            avg_loss_epoch = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {:.5f}".format(idx, avg_loss_epoch))



            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    current_loss_interval = (tr_loss - logging_loss) / (global_step - tr_nb + 1e-6)  # 防止除零

                    # 添加临时 acc 监控
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
                    preds = (probs >= 0.5)
                    labels_np = labels.detach().cpu().numpy()

                    logger.info("Step %s | Loss: %.5f", global_step, current_loss_interval)
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, modelencoder, eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))
                            # Save model checkpoint

                    
                    epoch_output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{idx}")
                    os.makedirs(epoch_output_dir, exist_ok=True)

                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save, os.path.join(epoch_output_dir, "model.pth"))
                    logger.info(f"Saved checkpoint for epoch {idx} to {epoch_output_dir}")

                    
                    
                    


# 新增参数modelencoder
def evaluate(args, model, tokenizer, modelencoder, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args, modelencoder, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0,
                                 pin_memory=True, collate_fn=collate_fn)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        # 从 batch dict 中提取并移到设备上(原来是元组，现在是字典)
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        position_idx = batch['position_idx'].to(args.device)
        label = batch['labels'].to(args.device)

        ast_adj = batch['ast_adj'].to(args.device)  # [B, M, M, 1]
        cfg_adj = batch['cfg_adj'].to(args.device)  # [B, M, M, 1]
        pdg_adj = batch['pdg_adj'].to(args.device)  # [B, M, M, 1]
        node_features = batch['node_features'].to(args.device)  # [B, M, F]
        node_mask = batch['node_mask'].to(args.device)  # [B, M]

        # 传参给模型的时候也要改
        with torch.no_grad():
            logit = model(input_ids, attention_mask, position_idx, None,
                          ast_adj, cfg_adj, pdg_adj,
                          node_features, node_mask)

            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logit, label.float())  # 自动 reduction='mean'
            eval_loss += loss.item()

        nb_eval_steps += 1
        logits.append(logit.cpu())
        labels.append(label.cpu())
    logits = torch.cat(logits, dim=0)  # [N, 1]
    labels = torch.cat(labels, dim=0)  # [N]

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()

    eval_acc = (preds == labels).float().mean().item()
    preds_np = preds.numpy()
    labels_np = labels.numpy()

    f1 = f1_score(labels_np, preds_np, zero_division=0)
    precision = precision_score(labels_np, preds_np, zero_division=0)
    recall = recall_score(labels_np, preds_np, zero_division=0)

    eval_loss = eval_loss / nb_eval_steps
            
    result = {
        "eval_loss": float(eval_loss),
        "eval_acc":round(eval_acc,4),
        'f1_score': round(f1, 4),
        'precision' :round(precision, 4),
        'recall': round(recall, 4),
    }
    return result


# 新增参数modelencoder
def test(args, model, tokenizer, modelencoder):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args, modelencoder, args.test_data_file)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    logits = []
    labels = []

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        # 从 batch dict 中提取并移到设备上(原来是元组，现在是字典)
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        position_idx = batch['position_idx'].to(args.device)
        label = batch['labels'].to(args.device)

        ast_adj = batch['ast_adj'].to(args.device)  # [B, M, M, 1]
        cfg_adj = batch['cfg_adj'].to(args.device)  # [B, M, M, 1]
        pdg_adj = batch['pdg_adj'].to(args.device)  # [B, M, M, 1]
        node_features = batch['node_features'].to(args.device)  # [B, M, F]
        node_mask = batch['node_mask'].to(args.device)  # [B, M]
        node_mask = batch['node_mask'].to(args.device)  # [B, M]

        # 传参给模型的时候也要改
        with torch.no_grad():
            logit = model(input_ids, attention_mask, position_idx, None,
                          ast_adj, cfg_adj, pdg_adj,
                          node_features, node_mask)

        logits.append(logit.cpu())
        labels.append(label.cpu())

    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
    eval_acc = (preds == labels).float().mean().item()
    labels_np = labels.numpy()
    preds_np = preds.numpy()

    precision = precision_score(labels_np, preds_np, zero_division=0)
    recall = recall_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, zero_division=0)

    metrics = {
        "test_accuracy": round(eval_acc, 4),
        "test_precision": round(precision, 4),
        "test_recall": round(recall, 4),
        "test_f1": round(f1, 4),
        "total_samples": len(labels)
    }

    # ========== 完善输出/记录逻辑 ==========
    # 1. 控制台打印（清晰排版）
    print("\n" + "=" * 50)
    print("***** Test Set Evaluation Results *****")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")

    # 2. 写入logger（和训练日志统一，方便后续查看）
    logger.info("***** Test Set Evaluation Results *****")
    for key, value in metrics.items():
        logger.info(f"  {key} = {value}")

    metrics_file = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Test metrics saved to {metrics_file}")

    preds_file = os.path.join(args.output_dir, "test_predictions.npz")
    np.savez(preds_file, labels=labels, preds=preds, logits=logits)
    logger.info(f"Test predictions saved to {preds_file}")
    # with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
    #     for example,pred in zip(eval_dataset.examples,preds):
    #         if pred:
    #             f.write(example.idx+'\t1\n')
    #         else:
    #             f.write(example.idx+'\t0\n')                    


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="../", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.02, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--eval_epoch", type=int, default=0, help="Which epoch checkpoint to evaluate/test.")

    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    device = torch.device("cuda")
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0


    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    # 先保留一下这个副本
    modelencoder = model
    model = GraphCodeBERT(model, config, tokenizer, args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache


        train_dataset = TextDataset(tokenizer, args, modelencoder, args.train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()

        # train本身虽然用不到，但是应该传过去，因为训练时验证是需要的：
        train(args, train_dataset, model, tokenizer, modelencoder)

    results = {}
    # 修复do_eval逻辑
    if args.do_eval and args.local_rank in [-1, 0]:
        model_path = os.path.join(args.output_dir, f"checkpoint-epoch-{args.eval_epoch}", "model.pth")
        logger.info(f"Loading model from {model_path}")
        model = torch.load(model_path)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, modelencoder, eval_when_training=False)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    # 修复do_test逻辑
    if args.do_test and args.local_rank in [-1, 0]:
        model_path = os.path.join(args.output_dir, f"checkpoint-epoch-{args.eval_epoch}", "model.pth")
        logger.info(f"Loading model from {model_path}")
        model = torch.load(model_path)
        model.to(args.device)
        # 新增参数modelencoder
        test(args, model, tokenizer, modelencoder)


    return results


if __name__ == "__main__":
    main()