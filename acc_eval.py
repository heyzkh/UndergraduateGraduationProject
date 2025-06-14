import json
import os
import re
from openai import OpenAI
from data_utils.data_utils import process_json_file


class DeepSeek:
    def __init__(self, api_key: str = "密钥",
                 base_url: str = "https://api.deepseek.com"):
        """
        初始化 DeepSeek。

        :param api_key: DeepSeek API 密钥
        :param base_url: DeepSeek API 的基础 URL，默认为 "https://api.deepseek.com"
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def call(
            self,
            prompt: str,
            model: str = "deepseek-chat",  # DeepSeek 的默认模型；deepseek-reasoner
            temperature: float = 1.,
    ):
        return self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": """
                你是一个常识关系验证系统。你的任务是根据人类常识知识，判断给定的头尾实体是否满足指定的关系。
                """},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )


def check_and_correct_tuple(entity_tuple, relationship, explain, deepseek):
    prompt = f"""
        下面是任务介绍：
        '{relationship}'的解释：{explain}
        思考判断[{entity_tuple}]是否符合'{relationship}'关系中的某一解释
        符合输出：1，不符合输出：0
    """

    #
    # 调用 DeepSeek API
    response = deepseek.call(prompt=prompt)

    if response is None:
        return None

    try:
        # 提取结果
        corrected_tuple = response.choices[0].message.content
        return corrected_tuple
    except Exception as e:
        print(f"处理响应结果时出错: {e}")
        return None


def process_json(input_file_path, output_file_path, explain, deepseek):

    data, relationship = process_json_file(input_file_path)
    print(f"{relationship} processing, explain: {explain}")
    # 处理每个实体元组
    res_data = []
    cnt_1 = 0
    label = 0
    for entity_tuple in data:
        try:
            text = check_and_correct_tuple(entity_tuple, relationship, explain, deepseek)
            # 获取修正意见中的实体元组
            # 使用正则表达式匹配列表
            if '1' in text:
                label = 1
                cnt_1 += 1
            elif '0' in text:
                label = 0
            else:
                label = 1
                cnt_1 += 1

        except Exception as e:
            print(f"处理 {entity_tuple} 时出错: {e}")
            label = 1
            cnt_1 += 1

        dic = {
            "entity_tuple": entity_tuple,
            "label": label
        }
        res_data.append(dic)
    # 将处理后的数据保存到新的 JSON 文件中
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(res_data, file, ensure_ascii=False, indent=4)
    print(f"数据处理完成，处理后的数据已保存到 {output_file_path}")

    tem_acc = {
        "relation": relationship,
        "acc": cnt_1 / len(data)
    }
    acc_res = []
    if os.path.exists("eval/acc_res.json"):
        with open("eval/acc_res.json", 'r', encoding='utf-8') as file:
            acc_res = json.load(file)
    acc_res.append(tem_acc)

    with open("eval/acc_res.json", 'w', encoding='utf-8') as file:
        json.dump(acc_res, file, ensure_ascii=False, indent=4)


def process_all_json_files(root_dir, deepseek):
    ent_tuple_input_file_path = ''
    output_file_path = ''
    explain = []

    # subdir是该子文件夹路径、dirs是改文件夹下的子文件夹的名称列表、files是该子文件夹下的文件
    for subdir, dirs, files in os.walk(root_dir):
        flag = 0
        for file in files:
            if file.endswith('ent_tuples.json'):
                ent_tuple_input_file_path = os.path.join(subdir, file)
                output_file_path = ent_tuple_input_file_path.replace('.json', '_eval.json')
                # print(f"ent_tuple_input_file_path: {ent_tuple_input_file_path}")
            if file.endswith('prompts.json'):
                explain = []
                prompt_file_path = os.path.join(subdir, file)
                # print(prompt_file_path)
                try:
                    # 打开并读取 JSON 文件
                    with open(prompt_file_path, 'r', encoding='utf-8') as f:
                        # 将 JSON 数据解析为 Python 对象
                        prompt = json.load(f)
                        if prompt:
                            # first_item = prompt[0]
                            # # 提取第一条数据的字符串部分（假设字符串部分是列表中的第一个元素）
                            # explain = first_item[0]
                            for p in prompt:
                                explain.append(p[0])
                except FileNotFoundError:
                    print(f"未找到文件: {prompt_file_path}")
                except json.JSONDecodeError:
                    print(f"无法解析 {prompt_file_path} 中的 JSON 数据，请检查文件格式。")
            else:
                flag = 1
                continue
        if flag == 1:
            process_json(ent_tuple_input_file_path, output_file_path, explain, deepseek)


deepseek = DeepSeek()
process_all_json_files("/workspace/bertnet/results/原论文数据top10_100", deepseek=deepseek)
# 可以说在搜索过程中加入r1的常识推理，将搜索出的实体对在r1中进行推理，对不合理的实体给出头实体进行启发，极大限度的对bert进行搜索，只进行一次启发，避免陷入死循环，因为bert对某些关系的理解并不擅长，使得搜索结果更准确。
# 最后的结果可以通过让大模型修改用于知识。