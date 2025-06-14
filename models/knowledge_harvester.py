# from tqdm import tqdm
# import numpy as np
# from scipy.special import softmax

# from models.language_model_wrapper import LanguageModelWrapper
# from models.entity_tuple_searcher import EntityTupleSearcher

# from data_utils.data_utils import fix_prompt_style, is_valid_prompt


# class KnowledgeHarvester:
#     def __init__(self,
#                  model_name,
#                  max_n_prompts=20,
#                  max_n_ent_tuples=10000,
#                  max_word_repeat=5,
#                  max_ent_subwords=1,
#                  prompt_temp=1.):
#         self._weighted_prompts = []
#         self._weighted_ent_tuples = []
#         self._max_n_prompts = max_n_prompts
#         self._max_n_ent_tuples = max_n_ent_tuples
#         self._max_word_repeat = max_word_repeat
#         self._max_ent_subwords = max_ent_subwords
#         self._prompt_temp = prompt_temp

#         self._model = LanguageModelWrapper(model_name=model_name)
#         self._ent_tuple_searcher = EntityTupleSearcher(model=self._model)

#         self._seed_ent_tuples = None

#     def clear(self):
#         self._weighted_prompts = []
#         self._weighted_ent_tuples = []
#         self._seed_ent_tuples = None

#     def set_seed_ent_tuples(self, seed_ent_tuples):
#         self._seed_ent_tuples = seed_ent_tuples

#     def set_prompts(self, prompts):
#         for prompt in prompts:
#             if is_valid_prompt(prompt=prompt):
#                 self._weighted_prompts.append([fix_prompt_style(prompt), 1.])

#     def update_prompts(self):
#         for i, (prompt, _) in enumerate(self._weighted_prompts):
#             pos_scores, neg_scores = [], []
#             for ent_tuple in self._seed_ent_tuples:
#                 ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]

#                 pos_scores.append(self.score(
#                     prompt=prompt, ent_tuple=ent_tuple))

#                 for ent_idx in range(len(ent_tuple)):
#                     for ent_tuple1 in self._seed_ent_tuples:
#                         if ent_tuple1[ent_idx] == ent_tuple[ent_idx]:
#                             continue

#                         ent_tuple_neg = \
#                             ent_tuple[:ent_idx] + \
#                             [ent_tuple1[ent_idx]] + \
#                             ent_tuple[ent_idx + 1:]

#                         neg_scores.append(self.score(
#                             prompt=prompt, ent_tuple=ent_tuple_neg))

#             pos_score = sum(pos_scores) / len(pos_scores)
#             neg_score = sum(neg_scores) / len(neg_scores)
#             print(f"Prompt: {prompt}, Pos Mean: {pos_score}, Neg Mean: {neg_score}")
#             epsilon = 1e-8  # 防除零极小值
#             # 动态计算负样本惩罚系数alpha（保留自适应逻辑）
#             # alpha = (pos_score + neg_score + 2) / (pos_score + neg_score + 4)
#             alpha = (pos_score - neg_score + 2) / (pos_score - neg_score + 4)

#             # 更新加权提示列表中当前提示的权重，计算公式为 (正例得分 - 0.5 * 负例得分) / 提示温度
#             # self._weighted_prompts[i][1] = \
#             #     (pos_score - alpha * neg_score) / self._prompt_temp
            
#             self._weighted_prompts[i][1] = (alpha * pos_score)  / self._prompt_temp

#         self._weighted_prompts = sorted(
#             self._weighted_prompts,
#             key=lambda t: t[1], reverse=True)[:self._max_n_prompts]

#         norm_weights = softmax([weight for _, weight in self._weighted_prompts])
#         norm_weights[norm_weights < 0.05] = 0.
#         norm_weights /= norm_weights.sum()

#         for i, norm_weight in enumerate(norm_weights):
#             self._weighted_prompts[i][1] = norm_weight
#         self._weighted_prompts = [
#             t for t in self._weighted_prompts if t[1] > 1e-4]

#     # def update_prompts(self):
#     #     # 静态温度参数（固定值，需在类初始化时定义）
#     #     tau = self._prompt_temp  # 例如 self.tau = 0.5
#     #     epsilon = 1e-8  # 防除零极小值

#     #     # 遍历所有提示
#     #     for i, (prompt, _) in enumerate(self._weighted_prompts):
#     #         pos_scores, neg_scores = [], []

#     #         # 计算正负样本得分（与原代码逻辑相同）
#     #         for ent_tuple in self._seed_ent_tuples:
#     #             ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]
#     #             pos_scores.append(self.score(prompt=prompt, ent_tuple=ent_tuple))

#     #             for ent_idx in range(len(ent_tuple)):
#     #                 for ent_tuple1 in self._seed_ent_tuples:
#     #                     if ent_tuple1[ent_idx] == ent_tuple[ent_idx]:
#     #                         continue
#     #                     ent_tuple_neg = ent_tuple[:ent_idx] + [ent_tuple1[ent_idx]] + ent_tuple[ent_idx + 1:]
#     #                     neg_scores.append(self.score(prompt=prompt, ent_tuple=ent_tuple_neg))

#     #         # 计算正负得分均值
#     #         # pos_mean = np.mean(pos_scores) if pos_scores else 0.0
#     #         # neg_mean = np.mean(neg_scores) if neg_scores else 0.0

#     #         # 计算正例得分的平均值
#     #         pos_mean = sum(pos_scores) / len(pos_scores)
#     #         # 计算负例得分的平均值
#     #         neg_mean = sum(neg_scores) / len(neg_scores)

#     #         # 动态计算负样本惩罚系数alpha（保留自适应逻辑）
#     #         alpha = pos_mean / (neg_mean + epsilon)

#     #         # 计算未归一化权重（使用静态tau）
#     #         raw_weight = (pos_mean - alpha * neg_mean) / tau  # 修改点
#     #         self._weighted_prompts[i][1] = raw_weight

#     #     # # Softmax归一化（数值稳定实现）
#     #     # weights = np.array([w for _, w in self._weighted_prompts])
#     #     # max_weight = np.max(weights)
#     #     # exp_weights = np.exp(weights - max_weight)
#     #     # norm_weights = exp_weights / exp_weights.sum()

#     #     # # 阈值过滤与再归一化（保留原逻辑）
#     #     # norm_weights[norm_weights < 0.05] = 0.
#     #     # norm_weights /= norm_weights.sum()

#     #     # # 更新权重并过滤微小值
#     #     # for i, norm_weight in enumerate(norm_weights):
#     #     #     self._weighted_prompts[i][1] = norm_weight
#     #     # self._weighted_prompts = [t for t in self._weighted_prompts if t[1] > 1e-4]

#     #     # # 排序与截断（保留原逻辑）
#     #     # self._weighted_prompts = sorted(
#     #     #     self._weighted_prompts,
#     #     #     key=lambda t: t[1],
#     #     #     reverse=True
#     #     # )[:self._max_n_prompts]

#     #             # 对加权提示列表按权重降序排序，并截取前 self._max_n_prompts 个元素
#     #     self._weighted_prompts = sorted(
#     #         self._weighted_prompts,
#     #         key=lambda t: t[1], reverse=True)[:self._max_n_prompts]

#     #     # 对加权提示列表中的权重进行 softmax 归一化
#     #     norm_weights = softmax([weight for _, weight in self._weighted_prompts])
#     #     # 将归一化后的权重中小于 0.05 的置为 0
#     #     norm_weights[norm_weights < 0.05] = 0.
#     #     # 将归一化后的权重重新归一化，使其总和为 1
#     #     norm_weights /= norm_weights.sum()

#     #     # 更新加权提示列表中的权重为归一化后的权重
#     #     for i, norm_weight in enumerate(norm_weights):
#     #         self._weighted_prompts[i][1] = norm_weight
#     #     # 过滤掉加权提示列表中权重小于 1e-4 的元素
#     #     self._weighted_prompts = [
#     #         t for t in self._weighted_prompts if t[1] > 1e-4]

#     def update_ent_tuples(self):
#         ent_tuples = self._ent_tuple_searcher.search(
#             weighted_prompts=self._weighted_prompts,
#             n=self._max_n_ent_tuples,
#             max_word_repeat=self._max_word_repeat,
#             max_ent_subwords=self._max_ent_subwords)

#         self._weighted_ent_tuples = []
#         for ent_tuple in tqdm(ent_tuples, desc='re-scoring ent_tuples'):
#             best_ent_tuple = None
#             best_score = float('-inf')
#             for t in range(1 << len(ent_tuple)):
#                 bin_code = f'{t:b}'
#                 bin_code = '0' * (len(ent_tuple) - len(bin_code)) + bin_code

#                 coded_ent_tuple = []
#                 for b, ent in zip(bin_code, ent_tuple):
#                     coded_ent_tuple.append(ent.title() if b == '1' else ent)

#                 score = self.score_ent_tuple(ent_tuple=coded_ent_tuple)
#                 if score > best_score:
#                     best_score = score
#                     best_ent_tuple = coded_ent_tuple

#             self._weighted_ent_tuples.append([best_ent_tuple, best_score])

#         self._weighted_ent_tuples = sorted(
#             self._weighted_ent_tuples, key=lambda t: t[1], reverse=True)

#         norm_weights = softmax(
#             [weight for _, weight in self._weighted_ent_tuples])
#         for i, norm_weight in enumerate(norm_weights):
#             self._weighted_ent_tuples[i][1] = norm_weight

#     def score_ent_tuple(self, ent_tuple):
#         score = 0.
#         for prompt, weight in self.weighted_prompts:
#             score += weight * self.score(prompt=prompt, ent_tuple=ent_tuple)

#         return score

#     def score(self, prompt, ent_tuple):
#         logprobs = self._model.fill_ent_tuple_in_prompt(
#             prompt=prompt, ent_tuple=ent_tuple)['mask_logprobs']

#         token_wise_score = sum(logprobs) / len(logprobs)
#         ent_wise_score = sum(logprobs) / len(ent_tuple)
#         min_score = min(logprobs)

#         return (token_wise_score + ent_wise_score + min_score) / 3.
#     @property
#     def weighted_ent_tuples(self):
#         return self._weighted_ent_tuples

#     @property
#     def weighted_prompts(self):
#         return self._weighted_prompts

from tqdm import tqdm
import numpy as np
from scipy.special import softmax

from models.language_model_wrapper import LanguageModelWrapper
from models.entity_tuple_searcher import EntityTupleSearcher

from data_utils.data_utils import fix_prompt_style, is_valid_prompt


class KnowledgeHarvester:
    def __init__(self,
                 model_name,
                 max_n_prompts=20,
                 max_n_ent_tuples=10000,
                 max_word_repeat=5,
                 max_ent_subwords=1,
                 prompt_temp=1.):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []
        self._max_n_prompts = max_n_prompts
        self._max_n_ent_tuples = max_n_ent_tuples
        self._max_word_repeat = max_word_repeat
        self._max_ent_subwords = max_ent_subwords
        self._prompt_temp = prompt_temp

        self._model = LanguageModelWrapper(model_name=model_name)
        self._ent_tuple_searcher = EntityTupleSearcher(model=self._model)

        self._seed_ent_tuples = None

    def clear(self):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []
        self._seed_ent_tuples = None

    def set_seed_ent_tuples(self, seed_ent_tuples):
        self._seed_ent_tuples = seed_ent_tuples

    def set_prompts(self, prompts):
        for prompt in prompts:
            if is_valid_prompt(prompt=prompt):
                self._weighted_prompts.append([fix_prompt_style(prompt), 1.])

    def update_prompts(self):
        for i, (prompt, _) in enumerate(self._weighted_prompts):
            pos_scores, neg_scores = [], []
            for ent_tuple in self._seed_ent_tuples:
                ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]

                pos_scores.append(self.score(
                    prompt=prompt, ent_tuple=ent_tuple))

                for ent_idx in range(len(ent_tuple)):
                    for ent_tuple1 in self._seed_ent_tuples:
                        if ent_tuple1[ent_idx] == ent_tuple[ent_idx]:
                            continue

                        ent_tuple_neg = \
                            ent_tuple[:ent_idx] + \
                            [ent_tuple1[ent_idx]] + \
                            ent_tuple[ent_idx + 1:]

                        neg_scores.append(self.score(
                            prompt=prompt, ent_tuple=ent_tuple_neg))

            pos_score = sum(pos_scores) / len(pos_scores)
            neg_score = sum(neg_scores) / len(neg_scores)
            print(f"Prompt: {prompt}, Pos Mean: {pos_score}, Neg Mean: {neg_score}")
            alpha = (pos_score - neg_score + 2) / (pos_score - neg_score + 4)
            
            # self._weighted_prompts[i][1] = (alpha * pos_score)  / self._prompt_temp

            self._weighted_prompts[i][1] = \
                (pos_score - 0.5 * neg_score) / self._prompt_temp

        self._weighted_prompts = sorted(
            self._weighted_prompts,
            key=lambda t: t[1], reverse=True)[:self._max_n_prompts]

        norm_weights = softmax([weight for _, weight in self._weighted_prompts])
        norm_weights[norm_weights < 0.05] = 0.
        norm_weights /= norm_weights.sum()

        for i, norm_weight in enumerate(norm_weights):
            self._weighted_prompts[i][1] = norm_weight
        self._weighted_prompts = [
            t for t in self._weighted_prompts if t[1] > 1e-4]

    def update_ent_tuples(self):
        ent_tuples = self._ent_tuple_searcher.search(
            weighted_prompts=self._weighted_prompts,
            n=self._max_n_ent_tuples,
            max_word_repeat=self._max_word_repeat,
            max_ent_subwords=self._max_ent_subwords)

        self._weighted_ent_tuples = []
        for ent_tuple in tqdm(ent_tuples, desc='re-scoring ent_tuples'):
            best_ent_tuple = None
            best_score = float('-inf')
            for t in range(1 << len(ent_tuple)):
                bin_code = f'{t:b}'
                bin_code = '0' * (len(ent_tuple) - len(bin_code)) + bin_code

                coded_ent_tuple = []
                for b, ent in zip(bin_code, ent_tuple):
                    coded_ent_tuple.append(ent.title() if b == '1' else ent)

                score = self.score_ent_tuple(ent_tuple=coded_ent_tuple)
                if score > best_score:
                    best_score = score
                    best_ent_tuple = coded_ent_tuple

            self._weighted_ent_tuples.append([best_ent_tuple, best_score])

        self._weighted_ent_tuples = sorted(
            self._weighted_ent_tuples, key=lambda t: t[1], reverse=True)

        norm_weights = softmax(
            [weight for _, weight in self._weighted_ent_tuples])
        for i, norm_weight in enumerate(norm_weights):
            self._weighted_ent_tuples[i][1] = norm_weight

    def score_ent_tuple(self, ent_tuple):
        score = 0.
        for prompt, weight in self.weighted_prompts:
            score += weight * self.score(prompt=prompt, ent_tuple=ent_tuple)

        return score

    def score(self, prompt, ent_tuple):
        logprobs = self._model.fill_ent_tuple_in_prompt(
            prompt=prompt, ent_tuple=ent_tuple)['mask_logprobs']

        token_wise_score = sum(logprobs) / len(logprobs)
        ent_wise_score = sum(logprobs) / len(ent_tuple)
        min_score = min(logprobs)

        return (token_wise_score + ent_wise_score + min_score) / 3.
    @property
    def weighted_ent_tuples(self):
        return self._weighted_ent_tuples

    @property
    def weighted_prompts(self):
        return self._weighted_prompts

