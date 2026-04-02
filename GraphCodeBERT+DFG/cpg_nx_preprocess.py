import networkx as nx
import pygraphviz as pgv
import numpy as np
import html

# 去除 HTML 实体转义，使 label 具有语义一致性
def normalize_label(label: str) -> str:
    return html.unescape(label)


# 提取节点类型
def parse_node_type(label: str) -> str:
    inner = label
    first_line = inner.split("<BR/>", 1)[0]
    node_type = first_line.split(",", 1)[0].strip()
    return node_type



# 提取节点代码
def parse_code(label: str, node_type: str) -> str:
    if "<BR/>" not in label:
        return ""

    inner = label
    parts = inner.split("<BR/>")

    content = parts[1:]

    if not content:
        return ""

    # IDENTIFIER / LITERAL /METHOD_REF 必须在中间
    if node_type in {"IDENTIFIER", "LITERAL", "METHOD_REF"}:
        return content[0].strip()

    # LOCAL 比较特殊，不到结尾
    if node_type == "LOCAL":
        # 提取 content 第一个元素并去除首尾空白
        local_content = content[0].strip()
        # 按:分割，取分割后的第一个部分（即:之前的内容）
        local_result = local_content.split(":")[0].strip()
        return local_result

    # 兜底
    return content[-1].strip()



def preprocess_cpg_sub(G, tokenizer, cpg_embeddings):

    # 先拿数据里的图
    G = G.cpg_object
    # 构建节点列表（以固定顺序！！！非常重要）
    nodes = list(G.nodes)

    max_nodes = 500  # 可以从 args 传入，或者固定值
    nodes = nodes[:max_nodes]

    # 构建节点极其索引（即{"joern的hash值": idx}，例如{"xxxx": 1, "xxx": 2}
    node2idx = {n: i for i, n in enumerate(nodes)}

    N = len(nodes)

    # 初始化三个邻接矩阵，大小一致，都是所有节点×所有节点
    A_ast = np.zeros((N, N), dtype=int)
    A_cfg = np.zeros((N, N), dtype=int)
    A_pdg = np.zeros((N, N), dtype=int)


    for u, v, k, data in G.edges(keys=True, data=True):
        if u not in node2idx or v not in node2idx:
            continue
        label = data.get("label", "")
        i, j = node2idx[u], node2idx[v]

        if label.startswith("AST"):
            A_ast[i, j] = 1
            A_ast[j, i] = 1
        elif label.startswith("CFG"):
            A_cfg[i, j] = 1
            A_cfg[j, i] = 1
        elif label.startswith("DDG"):
            A_pdg[i, j] = 1
            A_pdg[j, i] = 1
        elif label.startswith("CDG"):
            A_pdg[i, j] = 1
            A_pdg[j, i] = 1

    # 为三个邻接矩阵加自环
    np.fill_diagonal(A_ast, 1)
    np.fill_diagonal(A_cfg, 1)
    np.fill_diagonal(A_pdg, 1)

    # 提取节点特征（节点类型和节点代码）
    node_types = []
    node_features = []

    for n in nodes:
        raw_label = G.nodes[n].get("label", "")
        label = normalize_label(raw_label)

        node_type = parse_node_type(label)
        code = parse_code(label, node_type)

        #feat = encode_node(node_type, code)  # 你的 embedding / encoding
        #node_features.append(feat)
        node_types.append(node_type)
        node_features.append(code)


    # 核心嵌入逻辑如下：当前未考虑节点类型

    cpg_features = []
    for word in node_features:
        # 更合理的节点特征嵌入
        feature_id = tokenizer.encode(word, add_special_tokens=False)
        if len(feature_id) > 0:
            feature = cpg_embeddings[feature_id].mean(axis=0)
        else:
            feature = np.zeros(768)

        cpg_features.append(feature)
    X = np.stack(cpg_features)
    return A_ast, A_cfg, A_pdg, X