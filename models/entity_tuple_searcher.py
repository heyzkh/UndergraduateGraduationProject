# import string
# import torch
# import heapq

# from data_utils.data_utils import get_n_ents, get_mask_place, \
#     get_masked_prompt, get_n_masks, stopwords


# class EntityTupleSearcher:
#     def __init__(self, model):
#         self._model = model

#     def search(self, weighted_prompts, max_word_repeat, max_ent_subwords, n):
#         n_ents = get_n_ents(weighted_prompts[0][0])

#         collected_tuples_heap = []
#         repeat_cnt = {}

#         for t in range(max_ent_subwords ** n_ents):
#             n_masks = get_n_masks(
#                 t=t, n_ents=n_ents, max_ent_subwords=max_ent_subwords)
#             print(f'searching with n_masks={n_masks}')

#             self.dfs(
#                 weighted_prompts=weighted_prompts,
#                 n_ents=n_ents,
#                 n_masks=n_masks,
#                 cur_ent_tuple=[],
#                 cur_logprobs=[],
#                 collected_tuples_heap=collected_tuples_heap,
#                 repeat_cnt=repeat_cnt,
#                 max_word_repeat=max_word_repeat,
#                 n=n)

#         ent_tuples = sorted([t[1] for t in collected_tuples_heap])

#         ent_tuples = [ent_tuples[i] for i in range(len(ent_tuples))
#                       if i == 0 or ent_tuples[i] != ent_tuples[i - 1]]

#         return ent_tuples

#     def dfs(self,
#             weighted_prompts,
#             n_ents,
#             n_masks,
#             cur_ent_tuple,
#             cur_logprobs,
#             collected_tuples_heap,
#             repeat_cnt,
#             max_word_repeat,
#             n):
#         cur_ent_idx = len(cur_ent_tuple)

#         if cur_ent_idx == n_ents:
#             pred = [min(cur_logprobs), cur_ent_tuple]

#             for ent in cur_ent_tuple:
#                 for word in ent.split():
#                     if repeat_cnt.get(word, 0) + 1 > max_word_repeat:
#                         return

#             heapq.heappush(collected_tuples_heap, pred)
#             for ent in cur_ent_tuple:
#                 for word in ent.split():
#                     repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

#             while len(collected_tuples_heap) > n:
#                 heap_top = heapq.heappop(collected_tuples_heap)
#                 for ent in heap_top[1]:
#                     for word in ent.split():
#                         repeat_cnt[word] = repeat_cnt[word] - 1

#             return

#         collected_ents = []
#         logprob_threshold = float('-inf') if len(collected_tuples_heap) < n \
#             else collected_tuples_heap[0][0]

#         self.dfs_ent(
#             cur_ent_tuple=cur_ent_tuple,
#             n_masks=n_masks,
#             weighted_prompts=weighted_prompts,
#             cur_token_ids=[],
#             cur_logprobs=[],
#             collected_ent_heap=collected_ents,
#             logprob_threashold=logprob_threshold,
#             n=n if len(cur_ent_tuple) == 0 else max_word_repeat)

#         collected_ents.sort(reverse=True)

#         flag = set()
#         for ent_logprob, pred_ent in collected_ents:
#             if pred_ent in flag:
#                 continue
#             else:
#                 flag.add(pred_ent)

#             min_upd = min(cur_logprobs + [ent_logprob])
#             if len(collected_tuples_heap) == n and \
#                     min_upd < collected_tuples_heap[0][0]:
#                 break

#             weighted_prompts_upd = []
#             for prompt, weight in weighted_prompts:
#                 weighted_prompts_upd.append(
#                     [prompt.replace(f'<ENT{cur_ent_idx}>', pred_ent), weight])

#             self.dfs(
#                 weighted_prompts=weighted_prompts_upd,
#                 n_ents=n_ents,
#                 n_masks=n_masks,
#                 cur_ent_tuple=cur_ent_tuple + [pred_ent],
#                 cur_logprobs=cur_logprobs + [ent_logprob],
#                 collected_tuples_heap=collected_tuples_heap,
#                 repeat_cnt=repeat_cnt,
#                 max_word_repeat=max_word_repeat,
#                 n=n)
   
#     def _check_relation(self, ent_tuple):
#         h, t = ent_tuple
#         prompt = f"头实体 {h} 和尾实体 {t} 是否满足 {self.relation} 关系？回答1或0。"
#         response = self.deepseek.call(prompt)
#         return response.choices[0].message.content.strip() == '1'

#     def _generate_new_head(self, original_head):
#         prompt = f"Generate a new English head entity for relation {self.relation} replacing {original_head}:"
#         response = self.deepseek.call(prompt, temperature=1.2)
#         return response.choices[0].message.content.strip().lower()

#     def _add_to_heap(self, ent_tuple, logprobs, heap, repeat_cnt, max_repeat, n):
#         for ent in ent_tuple:
#             for word in ent.split():
#                 if repeat_cnt.get(word, 0) >= max_repeat:
#                     return
#         heapq.heappush(heap, [min(logprobs), ent_tuple])
#         for ent in ent_tuple:
#             for word in ent.split():
#                 repeat_cnt[word] = repeat_cnt.get(word, 0) + 1
#         while len(heap) > n:
#             removed = heapq.heappop(heap)
#             for ent in removed[1]:
#                 for word in ent.split():
#                     repeat_cnt[word] -= 1

#     def dfs_ent(self,
#                 cur_ent_tuple,
#                 n_masks,
#                 weighted_prompts,
#                 cur_token_ids,
#                 cur_logprobs,
#                 collected_ent_heap,
#                 logprob_threashold,
#                 n):
#         ent_idx = len(cur_ent_tuple)

#         if len(cur_token_ids) == n_masks[ent_idx]:
#             pred_ent = self._model.tokenizer.decode(cur_token_ids)

#             pred_ent = pred_ent.strip().lower()
#             # filter "the xxx", "new xxx", etc.
#             if any([word in stopwords for word in pred_ent.split()]):
#                 return

#             # filter entity with less than 3 characters
#             if len(pred_ent.replace(' ', '')) <= 2:
#                 return

#             # filter entity with single-character words
#             if min([len(t) for t in pred_ent.split()]) <= 1:
#                 return

#             # filter entity full of short words
#             if max([len(t) for t in pred_ent.split()]) <= 2:
#                 return

#             # filter entity with repeating words, e.g., "word word"
#             if len(pred_ent.split()) > 1 and len(set(pred_ent.split())) == 1:
#                 return

#             for ent in cur_ent_tuple:
#                 # filter repeating entity in the entity tuple,
#                 # e.g., "grassland" vs "grass land"
#                 if pred_ent.replace(' ', '') == ent.replace(' ', ''):
#                     return
#                 # filter repeating entity in the entity tuple,
#                 # e.g., "play" vs "playing"
#                 if ent.startswith(pred_ent) or pred_ent.startswith(ent):
#                     return

#             # filter entity appearing in the prompt
#             for raw_prompt, _ in weighted_prompts:
#                 if pred_ent in raw_prompt:
#                     return

#             heapq.heappush(collected_ent_heap, [min(cur_logprobs), pred_ent])
#             while len(collected_ent_heap) > n:
#                 heapq.heappop(collected_ent_heap)

#             return

#         mask_logits_total = None
#         for raw_prompt, weight in weighted_prompts:
#             prompt = raw_prompt.replace(
#                 f'<ENT{ent_idx}>',
#                 self._model.tokenizer.decode(cur_token_ids).lower() +
#                 self._model.tokenizer.mask_token * (
#                         n_masks[ent_idx] - len(cur_token_ids)))

#             input_text = get_masked_prompt(
#                 prompt=prompt, n_masks=n_masks,
#                 mask_token=self._model.tokenizer.mask_token)
#             mask_logits = self._model.get_mask_logits(input_text=input_text)

#             mask_idx_in_prompt = get_mask_place(
#                 ent_idx=ent_idx, n_masks=n_masks, prompt=raw_prompt)
#             mask_logits = mask_logits[mask_idx_in_prompt]

#             if mask_logits_total is None:
#                 mask_logits_total = torch.zeros_like(mask_logits)
#             mask_logits_total = mask_logits_total + mask_logits * weight

#         mask_logits_total = mask_logits_total / sum(
#             weight for _, weight in weighted_prompts)

#         mask_logits_total[self._model.banned_ids] = -float('inf')
#         logprobs = torch.log_softmax(mask_logits_total, dim=-1)
#         logprobs, pred_ids = torch.sort(logprobs, descending=True)

#         for logprob, pred_id in zip(logprobs, pred_ids):
#             min_logprob_upd = min(cur_logprobs + [logprob.item()])
#             if len(collected_ent_heap) == n and \
#                     min_logprob_upd < collected_ent_heap[0][0]:
#                 break

#             if min_logprob_upd < logprob_threashold:
#                 break

#             if not any([ch.isalpha() for ch in
#                         self._model.tokenizer.decode(pred_id)]):
#                 continue

#             if any([punc in self._model.tokenizer.decode(pred_id)
#                     for punc in string.punctuation]):
#                 continue

#             self.dfs_ent(
#                 cur_ent_tuple=cur_ent_tuple,
#                 n_masks=n_masks,
#                 weighted_prompts=weighted_prompts,
#                 cur_token_ids=cur_token_ids + [pred_id],
#                 cur_logprobs=cur_logprobs + [logprob.item()],
#                 collected_ent_heap=collected_ent_heap,
#                 logprob_threashold=logprob_threashold,
#                 n=n)
        
import string
import torch
import heapq
from models.deepseek import DeepSeek
from data_utils.data_utils import get_n_ents, get_mask_place, \
    get_masked_prompt, get_n_masks, stopwords
from collections import defaultdict

class EntityTupleSearcher:
    def __init__(self, model):
        self._model = model
        self.deepseek = DeepSeek()
        self.validated_pairs = set()  # 缓存已验证的实体对
        self.current_relation = None  # 由外部设置当前关系
        self.head_entity = []

    def search(self, weighted_prompts, max_word_repeat, max_ent_subwords, n):
        n_ents = get_n_ents(weighted_prompts[0][0])

        collected_tuples_heap = []
        repeat_cnt = {}

        for t in range(max_ent_subwords ** n_ents):
            n_masks = get_n_masks(
                t=t, n_ents=n_ents, max_ent_subwords=max_ent_subwords)
            print(f'searching with n_masks={n_masks}')

            self.dfs(
                weighted_prompts=weighted_prompts,
                n_ents=n_ents,
                n_masks=n_masks,
                cur_ent_tuple=[],
                cur_logprobs=[],
                collected_tuples_heap=collected_tuples_heap,
                repeat_cnt=repeat_cnt,
                max_word_repeat=max_word_repeat,
                n=n)

        ent_tuples = sorted([t[1] for t in collected_tuples_heap])

        ent_tuples = [ent_tuples[i] for i in range(len(ent_tuples))
                      if i == 0 or ent_tuples[i] != ent_tuples[i - 1]]
        
        # 在最终收集实体元组后添加验证步骤
        valid_ent_tuples = []
        for ent_tuple in ent_tuples:
            if tuple(ent_tuple) in self.validated_pairs:
                valid_ent_tuples.append(ent_tuple)
                continue
                
            if self._validate_ent_tuple(ent_tuple):
                valid_ent_tuples.append(ent_tuple)
                self.validated_pairs.add(tuple(ent_tuple))
            else:
                new_head = self._generate_new_head(ent_tuple)
                self.head_entity.append(new_head)
                # new_ent_tuple = [new_head, ent_tuple[1]]
                # 将新头实体加入搜索队列（示例实现）
                print(f'new head: {new_head} 搜索中...')
                valid_ent_tuples.extend(self._search_new_head(new_head, weighted_prompts, max_word_repeat, max_ent_subwords, n))
                print(f'new head: {new_head} 搜索完毕！！！')
        
        self.head_entity = []
        return valid_ent_tuples
    
    def _validate_ent_tuple(self, ent_tuple):
        head, tail = ent_tuple
        prompt = f"Verify if head entity '{head}' and tail entity '{tail}' satisfy {self.current_relation} relation. Respond with 1 if yes, 0 if no."
        response = self.deepseek.call(prompt)
        return '1' in response.choices[0].message.content
    
    def _generate_new_head(self, ent_tuple):
        _, tail = ent_tuple
        prompt = f"Generate a new English head entity that satisfies {self.current_relation} relation with tail entity '{tail}' and that the new head entity is not {self.head_entity}. Return only the head entity. No other redundant output."
        # prompt = f"Generate a new English head entity that satisfies {self.current_relation} relation with tail entity '{tail}'. Return only the head entity. No other redundant output."
        # prompt = f"Generate a commonly recognized head entity for '{self.current_relation}' relation that avoids logical/contextual connections to '{tail}'. Return only the head entity."
        response = self.deepseek.call(prompt, temperature=1.0)
        return response.choices[0].message.content.strip(' "\'').lower()
    
    def _search_new_head(self, new_head, weighted_prompts, max_word_repeat, max_ent_subwords, n):
        """基于新头实体生成符合当前关系的尾实体"""
        # 参数验证
        if not isinstance(new_head, str) or len(new_head) == 0:
            raise ValueError(f"Invalid head entity: {new_head}")
        if n < 1:
            raise ValueError(f"Invalid search count: {n}")
        
        # 初始化搜索参数
        n_ents = 1  # 只处理尾实体(ENT1)
        collected_tuples_heap = []
        repeat_cnt = defaultdict(int)  # 使用默认字典
        
        # 调整prompts格式 (固定ENT0为新头实体)
        adjusted_prompts = []
        for raw_prompt, weight in weighted_prompts:
            # 确保prompt包含ENT1占位符
            if '<ENT1>' not in raw_prompt:
                continue  # 跳过不适用模板
            
            # 构建新prompt示例: "Paris is located in <ENT1>"
            fixed_prompt = raw_prompt.replace('<ENT0>', new_head).replace('<ENT1>', self._model.tokenizer.mask_token)
            # print(f"fixed prompt: {fixed_prompt}")
            adjusted_prompts.append((fixed_prompt, weight))
        
        # 执行深度优先搜索
        for t in range(max_ent_subwords ** n_ents):
            
            self.dfs(
                weighted_prompts=adjusted_prompts,
                n_ents=n_ents,
                n_masks=[t+1],  # 确保传入整数列表
                cur_ent_tuple=[],
                cur_logprobs=[],
                collected_tuples_heap=collected_tuples_heap,
                repeat_cnt=repeat_cnt,
                max_word_repeat=max_word_repeat,
                n=10
            )
        
        # 后处理搜索结果
        valid_pairs = []
        seen_tails = set()
        # print(collected_tuples_heap)
        # 按概率排序并过滤重复项
        for score, (tail,) in sorted(collected_tuples_heap, key=lambda x: -x[0]):
            full_tuple = (new_head, tail)
            
            # # 基础过滤
            # if tail in seen_tails:
            #     continue
            # if len(tail) < 3:
                # continue
            
            # 关系验证
            # if self._validate_ent_tuple(full_tuple):
            #     valid_pairs.append(full_tuple)
            #     seen_tails.add(tail)
            #     if len(valid_pairs) >= n:
            #         break
            valid_pairs.append(full_tuple)
            break
        # 生成失败时的降级策略
        if not valid_pairs:
            print(f"Warning: No valid tails found for head {new_head}")
            return [(new_head, "unknown_tail")]  # 返回默认值保持结构
        
        return valid_pairs

    def dfs(self,
            weighted_prompts,
            n_ents,
            n_masks,
            cur_ent_tuple,
            cur_logprobs,
            collected_tuples_heap,
            repeat_cnt,
            max_word_repeat,
            n):
        cur_ent_idx = len(cur_ent_tuple)

        if cur_ent_idx == n_ents:
            pred = [min(cur_logprobs), cur_ent_tuple]

            for ent in cur_ent_tuple:
                for word in ent.split():
                    if repeat_cnt.get(word, 0) + 1 > max_word_repeat:
                        return

            heapq.heappush(collected_tuples_heap, pred)
            for ent in cur_ent_tuple:
                for word in ent.split():
                    repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

            while len(collected_tuples_heap) > n:
                heap_top = heapq.heappop(collected_tuples_heap)
                for ent in heap_top[1]:
                    for word in ent.split():
                        repeat_cnt[word] = repeat_cnt[word] - 1

            return

        collected_ents = []
        logprob_threshold = float('-inf') if len(collected_tuples_heap) < n \
            else collected_tuples_heap[0][0]

        self.dfs_ent(
            cur_ent_tuple=cur_ent_tuple,
            n_masks=n_masks,
            weighted_prompts=weighted_prompts,
            cur_token_ids=[],
            cur_logprobs=[],
            collected_ent_heap=collected_ents,
            logprob_threashold=logprob_threshold,
            n=n if len(cur_ent_tuple) == 0 else max_word_repeat)

        collected_ents.sort(reverse=True)

        flag = set()
        for ent_logprob, pred_ent in collected_ents:
            if pred_ent in flag:
                continue
            else:
                flag.add(pred_ent)

            min_upd = min(cur_logprobs + [ent_logprob])
            if len(collected_tuples_heap) == n and \
                    min_upd < collected_tuples_heap[0][0]:
                break

            weighted_prompts_upd = []
            for prompt, weight in weighted_prompts:
                weighted_prompts_upd.append(
                    [prompt.replace(f'<ENT{cur_ent_idx}>', pred_ent), weight])

            self.dfs(
                weighted_prompts=weighted_prompts_upd,
                n_ents=n_ents,
                n_masks=n_masks,
                cur_ent_tuple=cur_ent_tuple + [pred_ent],
                cur_logprobs=cur_logprobs + [ent_logprob],
                collected_tuples_heap=collected_tuples_heap,
                repeat_cnt=repeat_cnt,
                max_word_repeat=max_word_repeat,
                n=n)

    def dfs_ent(self,
                cur_ent_tuple,
                n_masks,
                weighted_prompts,
                cur_token_ids,
                cur_logprobs,
                collected_ent_heap,
                logprob_threashold,
                n):
        ent_idx = len(cur_ent_tuple)

        if len(cur_token_ids) == n_masks[ent_idx]:
            pred_ent = self._model.tokenizer.decode(cur_token_ids)

            pred_ent = pred_ent.strip().lower()
            # filter "the xxx", "new xxx", etc.
            if any([word in stopwords for word in pred_ent.split()]):
                return

            # filter entity with less than 3 characters
            if len(pred_ent.replace(' ', '')) <= 2:
                return

            # filter entity with single-character words
            if min([len(t) for t in pred_ent.split()]) <= 1:
                return

            # filter entity full of short words
            if max([len(t) for t in pred_ent.split()]) <= 2:
                return

            # filter entity with repeating words, e.g., "word word"
            if len(pred_ent.split()) > 1 and len(set(pred_ent.split())) == 1:
                return

            for ent in cur_ent_tuple:
                # filter repeating entity in the entity tuple,
                # e.g., "grassland" vs "grass land"
                if pred_ent.replace(' ', '') == ent.replace(' ', ''):
                    return
                # filter repeating entity in the entity tuple,
                # e.g., "play" vs "playing"
                if ent.startswith(pred_ent) or pred_ent.startswith(ent):
                    return

            # filter entity appearing in the prompt
            for raw_prompt, _ in weighted_prompts:
                if pred_ent in raw_prompt:
                    return

            heapq.heappush(collected_ent_heap, [min(cur_logprobs), pred_ent])
            while len(collected_ent_heap) > n:
                heapq.heappop(collected_ent_heap)

            return

        mask_logits_total = None
        for raw_prompt, weight in weighted_prompts:
            prompt = raw_prompt.replace(
                f'<ENT{ent_idx}>',
                self._model.tokenizer.decode(cur_token_ids).lower() +
                self._model.tokenizer.mask_token * (
                        n_masks[ent_idx] - len(cur_token_ids)))

            input_text = get_masked_prompt(
                prompt=prompt, n_masks=n_masks,
                mask_token=self._model.tokenizer.mask_token)
            mask_logits = self._model.get_mask_logits(input_text=input_text)

            mask_idx_in_prompt = get_mask_place(
                ent_idx=ent_idx, n_masks=n_masks, prompt=raw_prompt)
            mask_logits = mask_logits[mask_idx_in_prompt]

            if mask_logits_total is None:
                mask_logits_total = torch.zeros_like(mask_logits)
            mask_logits_total = mask_logits_total + mask_logits * weight

        mask_logits_total = mask_logits_total / sum(
            weight for _, weight in weighted_prompts)

        mask_logits_total[self._model.banned_ids] = -float('inf')
        logprobs = torch.log_softmax(mask_logits_total, dim=-1)
        logprobs, pred_ids = torch.sort(logprobs, descending=True)

        for logprob, pred_id in zip(logprobs, pred_ids):
            min_logprob_upd = min(cur_logprobs + [logprob.item()])
            if len(collected_ent_heap) == n and \
                    min_logprob_upd < collected_ent_heap[0][0]:
                break

            if min_logprob_upd < logprob_threashold:
                break

            if not any([ch.isalpha() for ch in
                        self._model.tokenizer.decode(pred_id)]):
                continue

            if any([punc in self._model.tokenizer.decode(pred_id)
                    for punc in string.punctuation]):
                continue

            self.dfs_ent(
                cur_ent_tuple=cur_ent_tuple,
                n_masks=n_masks,
                weighted_prompts=weighted_prompts,
                cur_token_ids=cur_token_ids + [pred_id],
                cur_logprobs=cur_logprobs + [logprob.item()],
                collected_ent_heap=collected_ent_heap,
                logprob_threashold=logprob_threashold,
                n=n)
        
