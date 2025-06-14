import re
from nltk.corpus import stopwords
import os
import json

stopwords = stopwords.words('english')
stopwords.extend([
    'everything', 'everybody', 'everyone',
    'anything', 'anybody', 'anyone',
    'something', 'somebody', 'someone',
    'nothing', 'nobody',
    'one', 'neither', 'either', 'many',
    'us', 'first', 'second', 'next',
    'following', 'last', 'new', 'main', 'also'])


def is_valid_prompt(prompt):
    for i in range(1, len(prompt)):
        if prompt[i:].startswith('<ENT') and prompt[i - 1] not in [' ', '\"']:
            return False

    return True


def get_n_ents(prompt):
    n = 0
    while f'<ENT{n}>' in prompt:
        n += 1
    return n


def get_sent(prompt, ent_tuple):
    sent = prompt
    for idx, ent in enumerate(ent_tuple):
        sent = sent.replace(f'<ENT{idx}>', ent)

    return sent


def get_mask_place(ent_idx, n_masks, prompt):
    mask_idx = 0
    for t in re.findall(r'<ENT[0-9]+>', prompt):
        t_idx = int(t[len('<ENT'):-1])
        if t_idx != ent_idx:
            mask_idx += n_masks[t_idx]
        else:
            break

    return mask_idx


def get_n_masks(t, n_ents, max_ent_subwords):
    n_masks = []
    for i in range(n_ents):
        n_masks.append(t % max_ent_subwords + 1)
        t //= max_ent_subwords

    return n_masks


def get_masked_prompt(prompt, n_masks, mask_token):
    input_text = prompt
    for ent_idx, n_mask in enumerate(n_masks):
        input_text = input_text.replace(f'<ENT{ent_idx}>', mask_token * n_mask)

    return input_text


def fix_prompt_style(prompt):
    prompt = prompt.strip(' .')
    if prompt[0].isalpha():
        prompt = prompt[0].upper() + prompt[1:]

    return prompt + ' .'


def find_sublist(a, b):
    for l in range(len(a)):
        if a[l:l+len(b)] == b:
            return l

    return None
    """
    增强版子列表查找，处理可能的tokenization差异
    """
    # sequence, sublist
    # sub_len = len(b)
    # for i in range(len(a) - sub_len + 1):
    #     match = True
    #     for j in range(sub_len):
    #         if a[i+j] != b[j]:
    #             match = False
    #             break
    #     if match:
    #         return i
    # # 容错机制：尝试匹配核心部分
    # if len(b) > 2:
    #     return find_sublist(a, b[1:-1])
    # return -1  # 改为返回-1而不是断言失败


def data_processing(data):
    # 去除小数部分，只保留包含字符串的子列表
    result = [item[0] for item in data]
    return result


def process_json_file(input_file_path):
    # 获取父类文件名，作为关系
    parent_dir = os.path.dirname(input_file_path)
    parent_dir_name = os.path.basename(parent_dir)
    # 读取 JSON 文件
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"未找到文件: {input_file_path}")
    # 处理数据
    processed_data = data_processing(data)

    return processed_data, parent_dir_name