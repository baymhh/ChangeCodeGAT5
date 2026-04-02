import argparse
import sys
import os

import multiprocessing
from multiprocessing import cpu_count, Manager, Pool, Queue
import subprocess
import time
from typing import cast

# 图形可视化模块（基于Graphviz），用于读取.dot格式的图形文件
import pygraphviz as pgv
# 序列化模块，用于将NetworkX图形对象保存为.pkl文件（方便后续加载使用）
import pickle
# 复杂网络处理模块，用于构建和操作有向图
import networkx as nx
import json
# 配置文件管理模块（OmegaConf），用于加载和解析.yaml/.yml配置文件
from omegaconf import DictConfig, OmegaConf




sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 从上级目录的configs模块中导入configure_arg_parser函数（用于构建配置文件的参数解析器）
from configs.parse_args import configure_arg_parser

# 获取当前机器的CPU核心数，赋值给USE_CPU（用于后续多进程配置参考）
USE_CPU = cpu_count()
# 获取当前脚本文件所在的目录路径
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# 获取当前脚本目录的父目录路径（用于定位数据集位置）
parent_directory = os.path.dirname(current_file_directory)


grandp_directory = os.path.dirname(parent_directory)




# 命令重试执行
def run_command_with_retries(command, max_retries=10):
    attempt = 0
    while attempt < max_retries:
        # 执行外部命令（shell=True表示使用系统shell执行命令）
        result = subprocess.run(command, shell=True)

        # 如果命令执行成功（返回码为0，绝大多数系统命令成功返回0）
        if result.returncode == 0:
            return result
        else:
            print('-' * 10)
            # 打印错误信息：命令内容、返回码、当前重试次数
            print(
                f"The command-{command}-execution failed, with a return code of {result.returncode}, and the attempt count is now {attempt + 1}")
            print('-' * 10)
            attempt += 1
            # 延时5秒后再次重试（避免频繁重试导致系统压力过大）
            time.sleep(5)
    print('-' * 10)
    print(f"The command-{command}-failed to execute multiple times, reaching the maximum number of attempts.")
    print('-' * 10)
    # 返回None，表示命令执行失败
    return None


def generate_dir(config: DictConfig, split_name, datasetName, generate_type):
    # 从配置对象中获取数据根目录名称
    data_dir = config.data_folder

    # 1. 构建临时代码文件目录路径：用于存放清洗后的临时代码片段
    #    os.path.join(...)：拼接多个路径为一个完整路径（跨系统兼容，自动处理/和\）

    # 这一步其实相当于在joern/data/文件夹下创建一个目录临时存放函数代码
    temp_function_dir_path = os.path.join(current_file_directory, data_dir, datasetName,
                                          f'{generate_type}_code_{split_name}')


    # 2. 构建Joern输出根目录路径：用于存放Joern解析和导出的中间文件

    # 这一步其实相当于在joern/data/文件夹下创建一个目录临时存放joern输出的二进制文件和dot文件
    output_dir_path = os.path.join(current_file_directory, data_dir, datasetName,
                                   f'{generate_type}_output_{split_name}')


    # 3. 构建最大体积.dot文件存放目录：用于存放筛选后的核心PDG可视化文件
    # 这一步其实相当于在joern/data/文件夹下创建一个目录存放核心dot
    output_dot_max_path = os.path.join(current_file_directory, data_dir, datasetName,
                                       f'{generate_type}_output_dot_{split_name}')

    # 4. 构建图形序列化文件目录：用于存放.pkl格式的NetworkX图形对象
    # 这一步其实相当于在joern/data/文件夹下创建一个目录存放.pkl格式的图对象
    output_pickle_path = os.path.join(current_file_directory, data_dir, datasetName,
                                      f'{generate_type}_output_pickle_{split_name}')


    # 清理已有目录（如果存在则删除，避免旧数据干扰）
    if os.path.exists(temp_function_dir_path):
        os.system(f'rm -rf {temp_function_dir_path}')
    if os.path.exists(output_dir_path):
        os.system(f'rm -rf {output_dir_path}')
    if os.path.exists(output_dot_max_path):
        os.system(f'rm -rf {output_dot_max_path}')
    if os.path.exists(output_pickle_path):
        os.system(f'rm -rf {output_pickle_path}')

    # 创建新目录（exist_ok=True表示如果目录已存在，不抛出异常，直接跳过）
    os.makedirs(temp_function_dir_path, exist_ok=True)
    os.makedirs(output_dir_path, exist_ok=True)
    os.makedirs(output_dot_max_path, exist_ok=True)
    os.makedirs(output_pickle_path, exist_ok=True)

    # 返回四个构建后的目录路径，供后续函数使用
    return temp_function_dir_path, output_dir_path, output_dot_max_path, output_pickle_path




# 生成程序依赖图（PDG）
def generate_pdg(config: DictConfig, js, clean_func, temp_function_dir_path, output_dir_path,
                 output_dot_max_path, output_pickle_path, export_format):
    try:
        # 从JSON数据中提取关键信息
        idx = js['idx']
        target = js['target']
        func = js['func']

        # 从配置中获取Joern工具的路径
        joern_path_parse = config.joern_path_parse    # Joern解析命令路径（生成CPG.bin）
        joern_path_export = config.joern_path_export  # Joern导出命令路径（从CPG.bin导出PDG）

        # 导出文件格式（如dot）
        export_type = config.export_type

        # 临时代码文件后缀（如.c/.java）
        generate_file_suffix = config.generate_file_suffix

        # 构建临时代码文件的名称（带ID和标签，避免重名）
        temp_function_file_name = f'temp_function-{idx}-{target}'
        temp_function_file_name_suffix = f'temp_function-{idx}-{target}{generate_file_suffix}'
        # 构建临时代码文件的完整路径
        temp_function_file_path = os.path.join(temp_function_dir_path, temp_function_file_name_suffix)

        # 构建最终输出的.dot文件名称
        final_output_dot_file_name = f'{export_format}-{idx}-{target}.{export_type}'

        # 将清洗后的代码写入临时文件（供Joern解析使用）
        with open(temp_function_file_path, 'w') as file:
            file.write(clean_func)

        # 1. 构建Joern解析命令：将临时代码文件解析为CPG（代码属性图）二进制文件
        cpg_file_path = os.path.join(output_dir_path, f'cpg-{temp_function_file_name}.bin')
        parse_command = f'{joern_path_parse} {temp_function_file_path} --output {cpg_file_path}'

        # 2. 构建Joern导出命令：将CPG文件导出为指定格式的子图文件
        output_dot = os.path.join(output_dir_path, f'out-{temp_function_file_name}')
        export_command = f'{joern_path_export} {cpg_file_path} --repr {export_format} --out {output_dot} --format {export_type}'

        # 执行解析命令（带重试机制）
        command_parse = f'{parse_command}'
        command_parse_return = run_command_with_retries(command_parse)
        # 如果解析失败，直接返回，终止当前函数
        if command_parse_return is None:
            print('-' * 10)
            print(f'Error: {parse_command} has problem')
            print('-' * 10)
            return

        # 执行导出命令（带重试机制）
        command_export = f'{export_command}'
        command_export_return = run_command_with_retries(command_export)
        # 如果导出失败，直接返回，终止当前函数
        if command_export_return is None:
            print('-' * 10)
            print(f'Error: {export_command} has problem')
            print('-' * 10)
            return

        # 3. 筛选导出目录中体积最大的.dot文件（Joern可能导出多个文件，最大的通常是完整PDG）
        output_dot_files = os.listdir(output_dot)
        # 计算每个文件的体积
        output_dot_files_size = [os.path.getsize(os.path.join(output_dot, file)) for file in output_dot_files]
        # 找到最大体积文件的索引
        output_dot_max_file_index = output_dot_files_size.index(max(output_dot_files_size))
        # 获取最大体积文件的名称
        output_dot_max_file_name = output_dot_files[output_dot_max_file_index]
        # 构建最大体积文件的完整路径
        output_dot_max_file_path = os.path.join(output_dot, output_dot_max_file_name)

        # 4. 过滤无效文件（体积为32字节的文件通常是空文件或无效文件，需跳过）
        if os.path.getsize(output_dot_max_file_path) != 32:
            # 构建目标文件路径（移动到指定目录）
            new_file_name_all_path = os.path.join(output_dot_max_path, final_output_dot_file_name)
            # 执行复制命令（将有效子图文件复制到目标目录）
            command_cp = f'cp {output_dot_max_file_path} {new_file_name_all_path}'
            command_cp_return = run_command_with_retries(command_cp)
            # 复制失败则终止
            if command_cp_return is None:
                print('-' * 10)
                print(f'Error: {command_cp} has problem')
                print('-' * 10)
                return

            # 5. 将.dot文件转换为NetworkX有向图，并添加元信息
            graph = pgv.AGraph(new_file_name_all_path)      # 用pygraphviz读取.dot文件


            #转为networkx形式时应该使用如下代码，自动识别为多重图
            G = nx.nx_agraph.from_agraph(graph)


            # 6. 将NetworkX图形对象序列化为.pkl文件（方便后续直接加载使用，无需重新解析）
            pickle_file_name = f'{export_format}-{idx}-{target}.pkl'
            pickle_file_path = os.path.join(output_pickle_path, pickle_file_name)
            pickle.dump(G, open(pickle_file_path, 'wb'))   # wb表示以二进制写入模式打开文件


        else:
            # 如果最大体积文件是32字节（无效文件），打印错误信息
            print(f'Error: {output_dot_max_file_path} has problem')
    except Exception as e:
        # 捕获所有异常，打印错误信息，避免单个任务失败导致整个进程池崩溃
        print(f'Exception Error: {e}')




# 数据集级批量处理函数
def readJSONDataAndGeneratePDG(config: DictConfig, datasetName, export_format):
    # 声明使用全局变量max_processes（进程池大小）
    global max_processes

    # 创建进程池，大小为max_processes（默认64），用于并行执行PDG生成任务
    pool = Pool(max_processes)
    # 打印当前机器CPU核心数
    print("CPU core num:", USE_CPU)
    # 数据集划分（测试集、验证集、训练集）
    split_name = ['test', 'valid', 'train']

    # 遍历每个数据集划分（test/valid/train）
    for splitName in split_name:
        # 生成当前划分对应的目录结构

        temp_function_dir_path, output_dir_path, output_dot_max_path, output_pickle_path = generate_dir(config,
                                                                                                        splitName,
                                                                                                        datasetName,
                                                                                                        export_format)
        # 打开当前划分的.jsonl文件（每行是一个JSON对象，存储代码数据）
        with open(os.path.join(grandp_directory, 'GraphCodeBERT+DFG', 'dataset', f'{splitName}.jsonl'), 'r') as f:

            # 逐行读取.jsonl文件

            # 这里暂时改为仅读取前100行
            all_lines = [line.strip() for line in f if line.strip()]
            all_lines = all_lines[:100]
            for line in all_lines:
                js = json.loads(line)  # 将每行字符串解析为JSON对象
                func = js['func']  # 提取原始代码函数

                clean_func = func

                # 进程池异步提交任务（非阻塞，进程池自动分配子进程执行）
                # 参数对应generate_pdg函数的入参
                pool.apply_async(generate_pdg,
                                 (config, js, clean_func, temp_function_dir_path, output_dir_path,
                                  output_dot_max_path, output_pickle_path, export_format))

    # 关闭进程池：不再接受新的任务提交
    pool.close()
    # 等待所有子进程任务执行完成：阻塞当前主线程，直到所有任务结束
    pool.join()


# 全局变量：数据集名称（后续通过命令行参数赋值）
dataset_name = None

# 全局变量：进程池最大进程数（默认64，可根据机器性能调整）
max_processes = 64


def main():
    print("-----------------BEGIN main-----------------")
    # 构建配置文件的参数解析器
    arg_parser = configure_arg_parser()
    # 解析命令行参数（unknown表示未识别的参数，不抛出异常）
    args, unknown = arg_parser.parse_known_args()
    # 加载配置文件，并强制转换为DictConfig类型
    config = cast(DictConfig, OmegaConf.load(args.config))
    global dataset_name   # 声明使用全局变量dataset_name

    # 获取配置中的导出格式列表（如pdg、cfg等，Joern支持的图形类型）
    export_format_list = list(config.joern.export_format)
    # 遍历每个导出格式，批量生成对应类型的图形
    for export_format in export_format_list:
        readJSONDataAndGeneratePDG(config.joern, dataset_name, export_format)

    print("---------------------END---------------------")


if __name__ == '__main__':

    # 创建命令行参数解析器（用于解析数据集名称和子项目名称）
    parser = argparse.ArgumentParser()

    # 添加数据集名称参数：-ds 或 --dataset_name，help是参数说明
    parser.add_argument('-ds', '--dataset_name', help='dataset_name')


    # 解析用户传入的命令行参数
    args = parser.parse_args()

    if args.dataset_name:
        dataset_name = args.dataset_name


    main()
