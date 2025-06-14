import re
import fire
import json
from nltk import sent_tokenize
from thefuzz import fuzz
from models.gpt4 import GPT4
from data_utils.data_utils import get_n_ents, get_sent, fix_prompt_style
from models.deepseek import DeepSeek
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


TRANSFORMATIONS_SENT = [['', ''], ['a ', ''], ['the ', '']]
TRANSFORMATIONS_ENT = [
    ['', ''], ['being', 'is'], ['being', 'are'], ['ing', ''], ['ing', 'e']]

SBERT = SentenceTransformer('all-MiniLM-L6-v2')


def get_paraphrase_prompt(model, prompt, ent_tuple):
    """
    使用 GPT-4-mini 生成改写后的提示。
    加入实体，让AI改写，在将实体去掉作为新提示
    :param prompt: 原始提示
    :param ent_tuple: 实体元组
    :return: 改写后的提示
    """
    assert get_n_ents(prompt) == len(ent_tuple)

    ent_tuple = [ent.lower() for ent in ent_tuple]
    sent = get_sent(prompt=prompt, ent_tuple=ent_tuple)
    paraphrase = f"""Please rephrase the sentence {sent} in a way that:
                    1. Strictly retains the original entities without alteration.
                    2. Preserves the core semantic relationship of {sent}.
                    3. Adopts a syntactic structure that is adaptable to diverse entity pairs .
                    4. Ensures the rewritten sentence is grammatically correct and naturally conveys the relationship.
                """
    for _ in range(5):
        # 我改的prompt
        # raw_response = model.call(prompt=f'{paraphrase}')
        # # 获取生成的文本
        # pattern = r'"([^"]*)"'
        # para_sent = raw_response.choices[0].message.content
        # match = re.search(pattern, para_sent)
        # if match:
        #     match = match.group(1)
        # else:
        #     continue
        # para_sent = match.strip().strip('.').lower()

        # 原来的prompt
        raw_response = model.call(prompt=f'paraphrase:\n{sent}\n')
        para_sent = raw_response.choices[0].message.content
        para_sent = sent_tokenize(para_sent)[0]
        para_sent = para_sent.strip().strip('.').lower()

        print('para_sent:', para_sent)

        prompt = para_sent
        valid = True
        for idx, ent in enumerate(ent_tuple):
            for trans_sent in TRANSFORMATIONS_SENT:
                for trans_ent in TRANSFORMATIONS_ENT:
                    if prompt.count(f'<ENT{idx}>') == 0:
                        transed_prompt = prompt.replace(*trans_sent)
                        transed_ent = ent.replace(*trans_ent)
                        if transed_prompt.count(transed_ent) == 1:
                            prompt = transed_prompt.replace(
                                transed_ent, f'<ENT{idx}>')

            if prompt.count(f'<ENT{idx}>') != 1:
                valid = False
                break

        if valid:
            return prompt

    return None


def search_prompts(init_prompts, seed_ent_tuples, similarity_threshold):
    """
    搜索并生成新的提示。

    :param init_prompts: 初始提示列表
    :param seed_ent_tuples: 种子实体元组列表
    :param similarity_threshold: 相似度阈值
    :return: 生成的提示列表
    """
    deepseek = DeepSeek()
    cache = {}
    prompts = []
    while True:
        new_prompts = []
        for prompt in init_prompts + init_prompts + prompts:
            for ent_tuple in seed_ent_tuples:
                ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]

                request_str = f'{prompt} ||| {ent_tuple}'
                if request_str not in cache or prompt in init_prompts:
                    cache[request_str] = get_paraphrase_prompt(model=deepseek,
                        prompt=prompt, ent_tuple=ent_tuple)

                para_prompt = cache[request_str]
                print(f'prompt: {prompt}\tent_tuple: {ent_tuple}'
                      f'\t-> para_prompt: {para_prompt}')

                if para_prompt is not None and \
                        para_prompt not in init_prompts + prompts:
                    new_prompts.append(para_prompt)

            if len(set(prompts + new_prompts)) >= 20:
                break

        if len(new_prompts) == 0:
            break
        else:
            flag = False
            for new_prompt in sorted(new_prompts, key=lambda t: len(t)):
                # if len(prompts) == 0:
                #     # 列表为空时直接添加新提示
                #     prompts.append(new_prompt)
                #     flag = True
                # else:
                #     max_sim = 0
                #     threshold = similarity_threshold
                #     for prompt in prompts:
                #         current_sim = fuzz.ratio(new_prompt, prompt)
                #         if current_sim >= threshold:
                #             # 发现超过阈值的相似度，立即终止循环
                #             max_sim = current_sim
                #             break
                #         if current_sim > max_sim:
                #             max_sim = current_sim
                #
                #     print(f'-- {new_prompt}: {max_sim}')
                #
                #     if max_sim < threshold:
                #         prompts.append(new_prompt)
                #         flag = True
                if len(prompts) == 0:
                    prompts.append(new_prompt)
                    flag = True
                else:
                    min_sim = 1
                    for prompt in prompts:
                        embeddings = SBERT.encode([new_prompt, prompt])
                        cos_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
                        if cos_sim < similarity_threshold:
                            min_sim = cos_sim
                            break
                        if cos_sim < min_sim:
                            min_sim = cos_sim

                    print(f'-- {new_prompt}: {min_sim}')

                    if min_sim >= similarity_threshold:
                        prompts.append(new_prompt)
                        flag = True

            prompts = list(set(prompts))
            prompts.sort(key=lambda s: len(s))

            if len(prompts) >= 10 or flag == False:
                break

    for i in range(len(prompts)):
        prompts[i] = fix_prompt_style(prompts[i])

    return prompts


def main(rel_set='conceptnet', similarity_threshold=0.6):
    """
    主函数，生成并保存提示。

    :param rel_set: 关系集名称
    :param similarity_threshold: 相似度阈值
    """
    relation_info = json.load(open(f'relation_info/{rel_set}.json'))

    for rel, info in relation_info.items():
        info['init_prompts'] = [
            fix_prompt_style(prompt) for prompt in info['init_prompts']]

        if 'prompts' not in info or len(info['prompts']) == 0:
            info['prompts'] = search_prompts(
                init_prompts=info['init_prompts'],
                seed_ent_tuples=info['seed_ent_tuples'],
                similarity_threshold=similarity_threshold)

            for key, value in info.items():
                print(f'{key}: {value}')
            for prompt in info['prompts']:
                print(f'- {prompt}')
            print('=' * 50)

        output_path = f'relation_info/{rel_set}.json'
        json.dump(relation_info, open(output_path, 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
