from allennlp.data.fields import *
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.nn.util import get_text_field_mask
from allennlp.data.tokenizers import Token
from allennlp.models import BasicClassifier, Model
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import F1Measure, Average, Metric
from allennlp.common.params import Params
from allennlp.commands.train import train_model
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.training.metrics.metric import Metric
from allennlp.nn import util

from typing import *
from overrides import overrides
import jieba
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet
import cv2 as cv
import os
import random
import re

torch.manual_seed(123)


def process_image(img, min_side=224):  # 等比例缩放与填充
    size = img.shape
    h, w = size[0], size[1]
    # 长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    # 下右填充
    top, bottom, left, right = 0, min_side-new_h, 0, min_side-new_w

    pad_img = cv.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right),
                                cv.BORDER_CONSTANT, value=[255,255,255]) # 从图像边界向上,下,左,右扩的像素数目

    return pad_img


def process_english_text(ori_text):
    ori_text = ori_text.replace('()', ' [Ans] ')

    text = re.split(r'([~⟂≅∥+-/=≠≈;,:.•?])', ori_text)
    # text = re.split('([~⟂≅∥+-\÷/=≠≈;,:.•?．\、\•\²])', ori_text)
    # text = re.split('([~⟂≅∥+-/=≠≈;,:.•?．]|[∠⊙△⁀▱])', ori_text)

    text = ' '.join(text)
    text = text.split()
    text = find_split_elements(text)

    text = ' '.join(text)
    text = text.replace('N_', ' N_')
    text = text.replace('{', ' ')
    text = text.replace('}', ' ')
    text = text.replace('′', '')
    text = text.replace('(', ' ( ')
    text = text.replace(')', ' ) ')

    # print('English problem, after processing: ', text)
    return text

# def add_mask(sentence):
#     tokens = sentence.split()
#     ori_tokens = tokens.copy()
#     mask_label = []
#
#     for i, token in enumerate(tokens):
#         prob = random.random()
#         if prob < 0.15:
#             prob /= 0.15
#
#             # 80% randomly change token to mask token
#             if prob < 0.8:
#                 tokens[i] = '[MASK]'
#
#             mask_label.append(1)
#
#         else:
#             mask_label.append(0)
#
#     return ori_tokens, tokens, mask_label
def add_mask(sentence):
    tokens = sentence.split()
    ori_tokens = tokens.copy()
    mask_label = []
    for i, token in enumerate(tokens):
        prob = random.random()
        if prob < 0.15:
            tokens[i] = '[MASK]'
            mask_label.append(1)
        else:
            mask_label.append(0)
    return ori_tokens, tokens, mask_label


def find_split_elements(sequence):
    new_sequence = []
    for token in sequence:
        if 'N_' in token or 'NS_' in token or 'frac' in token:
            new_sequence.append(token)
        elif token.istitle():  # 首字母大写，其余小写
            new_sequence.append(token)
        elif re.search(r'[A-Z]', token):
            tmp = list(token)
            new_sequence.extend(' '.join(tmp))
        else:
            new_sequence.append(token)

    return new_sequence




@DatasetReader.register("program_reader_english")
class SeqReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexer: Dict[str, TokenIndexer] = None,
                 target_token_indexer: Dict[str, TokenIndexer] = None,
                 model_name: str = None) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer
        self._source_token_indexer = source_token_indexer
        self._target_token_indexer = target_token_indexer
        self._model_name = model_name

        sub_dict_path = "data/sub_dataset_dict.pk"  # problems type
        with open(sub_dict_path, 'rb') as file:
            subset_dict = pickle.load(file)
        self.subset_dict = subset_dict

        self.all_points = ['切线', '垂径定理', '勾股定理', '同位角', '平行线', '三角形内角和', '三角形中位线', '平行四边形',
                      '相似三角形', '正方形', '圆周角', '直角三角形', '距离', '邻补角', '圆心角', '圆锥的计算', '三角函数',
                      '矩形', '旋转', '等腰三角形', '外接圆', '内错角', '菱形', '多边形', '对顶角', '三角形的外角', '角平分线',
                      '对称', '立体图形', '三视图', '圆内接四边形', '垂直平分线', '垂线', '扇形面积', '等边三角形', '平移',
                      '含30度角的直角三角形', '仰角', '三角形的外接圆与外心', '方向角', '坡角', '直角三角形斜边上的中线', '位似',
                      '平行线分线段成比例', '坐标与图形性质', '圆柱的计算', '俯角', '射影定理', '黄金分割', '钟面角']

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            for sample in dataset:
                yield self.text_to_instance(sample)

    @overrides
    def text_to_instance(self, sample) -> Instance:
        fields = {}

        image = sample['image']
        image = process_image(image)
        image = image/255
        img_rgb = np.zeros((3, image.shape[0], image.shape[1]))
        for i in range(3):
            img_rgb[i, :, :] = image
        fields['image'] = ArrayField(img_rgb)

        problem_with_space = process_english_text(sample['English_problem'])
        s_token = self._tokenizer.tokenize(problem_with_space)
        fields['source_tokens'] = TextField(s_token, self._source_token_indexer)

        t_token = self._tokenizer.tokenize(' '.join(sample['manual_program']))
        t_token.insert(0, Token(START_SYMBOL))
        t_token.append(Token(END_SYMBOL))
        fields['target_tokens'] = TextField(t_token, self._target_token_indexer)
        fields['source_nums'] = MetadataField(sample['numbers'])
        fields['choice_nums'] = MetadataField(sample['choice_nums'])
        fields['label'] = MetadataField(sample['label'])

        type = self.subset_dict[sample['id']]
        fields['type'] = MetadataField(type)
        fields['data_id'] = MetadataField(sample['id'])

        point_label = np.zeros(50, np.float32)
        exam_points = sample['formal_point']
        for point in exam_points:
            point_id = self.all_points.index(point)
            point_label[point_id] = 1
        fields['point_label'] = ArrayField(np.array(point_label))

        return Instance(fields)


@DatasetReader.register("s2s_manual_reader")
class SeqReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexer: Dict[str, TokenIndexer] = None,
                 target_token_indexer: Dict[str, TokenIndexer] = None,
                 model_name: str = None) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer
        self._source_token_indexer = source_token_indexer
        self._target_token_indexer = target_token_indexer
        self._model_name = model_name

        sub_dict_path = "data/sub_dataset_dict.pk"  # problems type
        with open(sub_dict_path, 'rb') as file:
            subset_dict = pickle.load(file)
        self.subset_dict = subset_dict

        self.all_points = ['切线', '垂径定理', '勾股定理', '同位角', '平行线', '三角形内角和', '三角形中位线', '平行四边形',
                      '相似三角形', '正方形', '圆周角', '直角三角形', '距离', '邻补角', '圆心角', '圆锥的计算', '三角函数',
                      '矩形', '旋转', '等腰三角形', '外接圆', '内错角', '菱形', '多边形', '对顶角', '三角形的外角', '角平分线',
                      '对称', '立体图形', '三视图', '圆内接四边形', '垂直平分线', '垂线', '扇形面积', '等边三角形', '平移',
                      '含30度角的直角三角形', '仰角', '三角形的外接圆与外心', '方向角', '坡角', '直角三角形斜边上的中线', '位似',
                      '平行线分线段成比例', '坐标与图形性质', '圆柱的计算', '俯角', '射影定理', '黄金分割', '钟面角']

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            for sample in dataset:
                yield self.text_to_instance(sample)

    @overrides
    def text_to_instance(self, sample) -> Instance:
        fields = {}

        image = sample['image']
        image = process_image(image)
        image = image/255
        img_rgb = np.zeros((3, image.shape[0], image.shape[1]))
        for i in range(3):
            img_rgb[i, :, :] = image
        fields['image'] = ArrayField(img_rgb)

        s_token = self._tokenizer.tokenize(' '.join(sample['token_list']))
        fields['source_tokens'] = TextField(s_token, self._source_token_indexer)
        t_token = self._tokenizer.tokenize(' '.join(sample['manual_program']))
        t_token.insert(0, Token(START_SYMBOL))
        t_token.append(Token(END_SYMBOL))
        fields['target_tokens'] = TextField(t_token, self._target_token_indexer)
        fields['source_nums'] = MetadataField(sample['numbers'])
        fields['choice_nums'] = MetadataField(sample['choice_nums'])
        fields['label'] = MetadataField(sample['label'])

        type = self.subset_dict[sample['id']]
        fields['type'] = MetadataField(type)
        fields['data_id'] = MetadataField(sample['id'])

        equ_list = []

        equ = sample['manual_program']
        equ_token = self._tokenizer.tokenize(' '.join(equ))
        equ_token.insert(0, Token(START_SYMBOL))
        equ_token.append(Token(END_SYMBOL))
        equ_token = TextField(equ_token, self._source_token_indexer)
        equ_list.append(equ_token)

        # TODO: delete ?
        fields['equ_list'] = ListField(equ_list)
        fields['manual_program'] = MetadataField(sample['manual_program'])

        point_label = np.zeros(50, np.float32)
        exam_points = sample['formal_point']
        for point in exam_points:
            point_id = self.all_points.index(point)
            point_label[point_id] = 1
        fields['point_label'] = ArrayField(np.array(point_label))

        return Instance(fields)


# V6: after acl, fix mask
@DatasetReader.register("pretrain_reader_english")
class SeqReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexer: Dict[str, TokenIndexer] = None,
                 target_token_indexer: Dict[str, TokenIndexer] = None,
                 model_name: str = None) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer
        self._source_token_indexer = source_token_indexer
        self._target_token_indexer = target_token_indexer
        self._model_name = model_name

    @overrides
    def _read(self, file_paths):
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                dataset = pickle.load(f)
                for sample in dataset:
                    print('-------------------------------')
                    if 'problem_form' not in sample.keys():
                        yield self.calculation_to_instance(sample)
                    else:
                        yield self.proving_to_instance(sample)

    def process_proving_English_input(self, ori_text):

        text = re.split('([~⟂≅∥+-/=]|bisects|and)', ori_text)
        text = ' '.join(text)
        text = text.replace('m∠', '∠')
        text = text.split()

        text = find_split_elements(text)
        text = ' '.join(text)

        return text

    def process_Chinese_solving(self, ori_text):

        pattern = re.compile(r'([+=-]|π|×|\\frac|\\sqrt|\√|[\∵\∴\∽\≌\⊥\/]|[\.,\，\：\；\中]|\d+\.?\d*|)')
        text = re.split(pattern, ori_text)

        # tokenize
        # delete_list = ['', '^{°}', '{', '}', '°', 'cm', 'm', '米']
        delete_list = ['', '^{°}', '{', '}', '°', 'cm', 'm', '米', ',']
        rebudant_list = ['解：', '故选']
        text2 = []
        for t in text:
            append_flag = True
            for d in delete_list:
                if d in t:
                    t = t.replace(d, '')
            if t in delete_list:
                append_flag = False
            for rebundant in rebudant_list:
                if rebundant in t:
                    append_flag = False

            if append_flag:
                text2.append(t)

        zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
        text3 = []
        for t in text2:
            t = re.sub(zh_pattern, ' ', t)
            text3.append(t)

        # combine
        text4 = ' '.join(text3)
        text4 = text4.replace('/ /', '//')
        text4 = text4.replace('∠ ', '∠')
        text4 = text4.replace('△ ', '△')
        text4 = text4.replace(r'\frac 1 2', ' Half ')

        replace_list = [':', '.', '．', '、', '′', '~', '″', '【', '】']
        for r in replace_list:
            text4 = text4.replace(r, '')

        text4 = text4.replace('(', ' ( ')
        text4 = text4.replace(')', ' ) ')
        text4 = text4.split()

        # split elements: ABC -> A B C
        text4 = find_split_elements(text4)

        # store variable
        text5 = []
        elements = []
        nums = []

        # replace only nums
        for t in text4:
            if re.search(r'\∠', t):
                text5.append(t)
            elif re.search(r'\d', t):  # NS: number in solving
                if float(t) in nums:
                    text5.append('NS_'+str(nums.index(float(t))))
                else:
                    text5.append('NS_'+str(len(nums)))
                    nums.append(float(t))
            else:
                text5.append(t)
        elements = None

        text5 = ' '.join(text5)
        text5 = text5.split()

        print('after process solving', len(text5), elements, nums, text5)

        return text5, elements, nums


    def calculation_to_instance(self, sample) -> Instance:
        fields = {}

        image = sample['image']
        image = process_image(image)
        image = image/255
        img_rgb = np.zeros((3, image.shape[0], image.shape[1]))
        for i in range(3):
            img_rgb[i, :, :] = image
        fields['image'] = ArrayField(img_rgb)

        ori_solving = sample['answer']
        solving, solving_ele, solving_nums = self.process_Chinese_solving(ori_solving)

        problem_with_space = process_english_text(sample['English_problem'])
        # text = problem_with_space.split()

        # TODO: add mask
        problem_with_space = 'Calculation:  ' + problem_with_space
        text, masked_text, mask_label = add_mask(problem_with_space)
        mask_label = np.array(mask_label)
        # NLU mask
        masked_text = ' '.join(masked_text)
        # masked_text = 'Calculation: ' + masked_text
        masked_text = self._tokenizer.tokenize(masked_text)
        print('calculation masked\n', masked_text)
        fields['masked_text'] = TextField(masked_text, self._source_token_indexer)
        fields['mask_label'] = ArrayField(mask_label)

        # whole
        s_token = text
        t_token = solving

        s_token = ' '.join(s_token)
        t_token = ' '.join(t_token)

        # s_token = 'Calculation: ' + s_token

        print('calculation source\n', s_token)
        s_token = self._tokenizer.tokenize(s_token)
        t_token = self._tokenizer.tokenize(t_token)

        t_token.insert(0, Token(START_SYMBOL))
        t_token.append(Token(END_SYMBOL))

        fields['source_tokens'] = TextField(s_token, self._source_token_indexer)
        fields['target_tokens'] = TextField(t_token, self._target_token_indexer)

        fields['source_nums'] = MetadataField(sample['numbers'])
        fields['choice_nums'] = MetadataField(sample['choice_nums'])
        fields['label'] = MetadataField(sample['label'])

        fields['problem_form'] = MetadataField(None)

        return Instance(fields)

    def proving_to_instance(self, sample) -> Instance:
        fields = {}

        image = sample['img']
        image = process_image(image)
        image = image/255
        image = image.transpose(2, 0, 1)
        fields['image'] = ArrayField(image)

        # add elements to input
        sample['question'] += ' (Elements: '
        for element in sample['elements']:
            sample['question'] = sample['question'] + ' ' + element + ' , '
        sample['question'] += ')'

        question = self.process_proving_English_input(sample['question'])

        # add prefix
        question = 'Proving: ' + question

        print('proving\n', question)
        s_token = self._tokenizer.tokenize(question)
        # s_token = self._tokenizer.tokenize(' '.join(sample['question']))

        fields['source_tokens'] = TextField(s_token, self._source_token_indexer)

        t_token = self._tokenizer.tokenize(' '.join(sample['proving_sequence']))
        t_token.insert(0, Token(START_SYMBOL))
        t_token.append(Token(END_SYMBOL))
        fields['target_tokens'] = TextField(t_token, self._target_token_indexer)

        fields['source_nums'] = MetadataField(None)
        # fields['answer'] = MetadataField(None)

        fields['problem_form'] = MetadataField(sample['problem_form'])
        # fields['problem_type'] = MetadataField(sample['problem_type'])
        # fields['reasoning_skill'] = MetadataField(sample['reasoning_skill'])

        fields['source_nums'] = MetadataField(None)
        fields['choice_nums'] = MetadataField(None)
        fields['label'] = MetadataField(None)


        return Instance(fields)


# # V5: split elements, ABC -> A B C
# @DatasetReader.register("pretrain_reader_english")
# class SeqReader(DatasetReader):
#     def __init__(self,
#                  tokenizer: Tokenizer = None,
#                  source_token_indexer: Dict[str, TokenIndexer] = None,
#                  target_token_indexer: Dict[str, TokenIndexer] = None,
#                  model_name: str = None) -> None:
#         super().__init__(lazy=False)
#         self._tokenizer = tokenizer
#         self._source_token_indexer = source_token_indexer
#         self._target_token_indexer = target_token_indexer
#         self._model_name = model_name
#
#     @overrides
#     def _read(self, file_paths):
#         for file_path in file_paths:
#             with open(file_path, 'rb') as f:
#                 dataset = pickle.load(f)
#                 for sample in dataset:
#                     print('-------------------------------')
#                     if 'problem_form' not in sample.keys():
#                         yield self.calculation_to_instance(sample)
#                     else:
#                         yield self.proving_to_instance(sample)
#
#     def process_proving_English_input(self, ori_text):
#
#         text = re.split('([~⟂≅∥+-/=]|bisects|and)', ori_text)
#         text = ' '.join(text)
#         text = text.replace('m∠', '∠')
#         text = text.split()
#
#         text = find_split_elements(text)
#         text = ' '.join(text)
#
#         return text
#
#     def process_Chinese_solving(self, ori_text):
#
#         pattern = re.compile(r'([+=-]|π|×|\\frac|\\sqrt|\√|[\∵\∴\∽\≌\⊥\/]|[\.,\，\：\；\中]|\d+\.?\d*|)')
#         text = re.split(pattern, ori_text)
#
#         # tokenize
#         # delete_list = ['', '^{°}', '{', '}', '°', 'cm', 'm', '米']
#         delete_list = ['', '^{°}', '{', '}', '°', 'cm', 'm', '米', ',']
#         rebudant_list = ['解：', '故选']
#         text2 = []
#         for t in text:
#             append_flag = True
#             for d in delete_list:
#                 if d in t:
#                     t = t.replace(d, '')
#             if t in delete_list:
#                 append_flag = False
#             for rebundant in rebudant_list:
#                 if rebundant in t:
#                     append_flag = False
#
#             if append_flag:
#                 text2.append(t)
#
#         zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
#         text3 = []
#         for t in text2:
#             t = re.sub(zh_pattern, ' ', t)
#             text3.append(t)
#
#         # combine
#         text4 = ' '.join(text3)
#         text4 = text4.replace('/ /', '//')
#         text4 = text4.replace('∠ ', '∠')
#         text4 = text4.replace('△ ', '△')
#         text4 = text4.replace(r'\frac 1 2', ' Half ')
#
#         replace_list = [':', '.', '．', '、', '′', '~', '″', '【', '】']
#         for r in replace_list:
#             text4 = text4.replace(r, '')
#
#         text4 = text4.replace('(', ' ( ')
#         text4 = text4.replace(')', ' ) ')
#         text4 = text4.split()
#
#         # split elements: ABC -> A B C
#         text4 = find_split_elements(text4)
#
#         # store variable
#         text5 = []
#         elements = []
#         nums = []
#
#         # replace only nums
#         for t in text4:
#             if re.search(r'\∠', t):
#                 text5.append(t)
#             elif re.search(r'\d', t):  # NS: number in solving
#                 if float(t) in nums:
#                     text5.append('NS_'+str(nums.index(float(t))))
#                 else:
#                     text5.append('NS_'+str(len(nums)))
#                     nums.append(float(t))
#             else:
#                 text5.append(t)
#         elements = None
#
#         text5 = ' '.join(text5)
#         text5 = text5.split()
#
#         print('after process solving', len(text5), elements, nums, text5)
#
#         return text5, elements, nums
#
#
#     def calculation_to_instance(self, sample) -> Instance:
#         fields = {}
#
#         image = sample['image']
#         image = process_image(image)
#         image = image/255
#         img_rgb = np.zeros((3, image.shape[0], image.shape[1]))
#         for i in range(3):
#             img_rgb[i, :, :] = image
#         fields['image'] = ArrayField(img_rgb)
#
#         ori_solving = sample['answer']
#         solving, solving_ele, solving_nums = self.process_Chinese_solving(ori_solving)
#
#         problem_with_space = process_english_text(sample['English_problem'])
#         text = problem_with_space.split()
#
#         # TODO: add mask
#         # text, masked_text, mask_label = add_mask(problem_with_space)
#         # mask_label = np.array(mask_label)
#
#         # whole
#         s_token = text
#         t_token = solving
#
#         # random mask
#         # r1 = random.random()
#         # r2 = random.random()
#         # mask_start = int(min(r1, r2)*len(solving))
#         # mask_end = max(int(max(r1, r2)*len(solving)), mask_start+1)
#         # t_token = solving[mask_start:mask_end]
#         # numbers = [str(num) for num in sample['numbers']]
#         # s_token = text + solving[:mask_start] + ['[MASK]'] + solving[mask_end:] #+ ['Number:'] + numbers
#
#         # part mask
#         # L = 1
#         # r1 = random.random()
#         # mask_start = min(int(r1*len(solving)), len(solving)-L)
#         # mask_end = min(mask_start+L, len(solving))
#         # t_token = solving[mask_start:mask_end]
#         # s_token = text + solving[:mask_start] + ['[MASK]'] + solving[mask_end:]
#
#         s_token = ' '.join(s_token)
#         t_token = ' '.join(t_token)
#
#         s_token = 'Calculation: ' + s_token
#
#         print('calculation\n', s_token)
#         s_token = self._tokenizer.tokenize(s_token)
#         t_token = self._tokenizer.tokenize(t_token)
#
#         t_token.insert(0, Token(START_SYMBOL))
#         t_token.append(Token(END_SYMBOL))
#
#         fields['source_tokens'] = TextField(s_token, self._source_token_indexer)
#         fields['target_tokens'] = TextField(t_token, self._target_token_indexer)
#
#         fields['source_nums'] = MetadataField(sample['numbers'])
#         fields['choice_nums'] = MetadataField(sample['choice_nums'])
#         fields['label'] = MetadataField(sample['label'])
#
#         # NLU mask
#         # masked_text = ' '.join(masked_text)
#         # masked_text = self._tokenizer.tokenize(masked_text)
#         # fields['masked_text'] = TextField(masked_text, self._source_token_indexer)
#         # fields['mask_label'] = ArrayField(mask_label)
#
#         fields['problem_form'] = MetadataField(None)
#
#         return Instance(fields)
#
#     def proving_to_instance(self, sample) -> Instance:
#         fields = {}
#
#         image = sample['img']
#         image = process_image(image)
#         image = image/255
#         image = image.transpose(2, 0, 1)
#         fields['image'] = ArrayField(image)
#
#         # add elements to input
#         sample['question'] += ' (Elements: '
#         for element in sample['elements']:
#             sample['question'] = sample['question'] + ' ' + element + ' , '
#         sample['question'] += ')'
#
#         question = self.process_proving_English_input(sample['question'])
#
#         # add prefix
#         question = 'Proving: ' + question
#
#         print('proving\n', question)
#         s_token = self._tokenizer.tokenize(question)
#         # s_token = self._tokenizer.tokenize(' '.join(sample['question']))
#
#         fields['source_tokens'] = TextField(s_token, self._source_token_indexer)
#
#         t_token = self._tokenizer.tokenize(' '.join(sample['proving_sequence']))
#         t_token.insert(0, Token(START_SYMBOL))
#         t_token.append(Token(END_SYMBOL))
#         fields['target_tokens'] = TextField(t_token, self._target_token_indexer)
#
#         fields['source_nums'] = MetadataField(None)
#         # fields['answer'] = MetadataField(None)
#
#         fields['problem_form'] = MetadataField(sample['problem_form'])
#         # fields['problem_type'] = MetadataField(sample['problem_type'])
#         # fields['reasoning_skill'] = MetadataField(sample['reasoning_skill'])
#
#         fields['source_nums'] = MetadataField(None)
#         fields['choice_nums'] = MetadataField(None)
#         fields['label'] = MetadataField(None)
#
#
#         return Instance(fields)


@DatasetReader.register("Uni_reader_english")
class SeqReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexer: Dict[str, TokenIndexer] = None,
                 target_token_indexer: Dict[str, TokenIndexer] = None,
                 model_name: str = None) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer
        self._source_token_indexer = source_token_indexer
        self._target_token_indexer = target_token_indexer
        self._model_name = model_name

        sub_dict_path = "data/sub_dataset_dict.pk"  # problems type
        with open(sub_dict_path, 'rb') as file:
            subset_dict = pickle.load(file)
        self.subset_dict = subset_dict

    @overrides
    def _read(self, file_paths):
        print(file_paths)
        if isinstance(file_paths, str):
            file_paths = file_paths.split(',')

        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                dataset = pickle.load(f)
                for sample in dataset:
                    print('-------------------------------')
                    if 'problem_form' not in sample.keys():
                        yield self.calculation_to_instance(sample)
                    else:
                        yield self.proving_to_instance(sample)

    def process_proving_English_input(self, ori_text):

        text = re.split('([~⟂≅∥+-/=]|bisects|and)', ori_text)
        text = ' '.join(text)
        text = text.replace('m∠', '∠')
        text = text.split()

        text = find_split_elements(text)
        text = ' '.join(text)

        return text

    def calculation_to_instance(self, sample) -> Instance:
        fields = {}

        image = sample['image']
        image = process_image(image)
        image = image/255
        img_rgb = np.zeros((3, image.shape[0], image.shape[1]))
        for i in range(3):
            img_rgb[i, :, :] = image
        fields['image'] = ArrayField(img_rgb)

        problem_with_space = process_english_text(sample['English_problem'])
        s_token = 'Calculation: ' + problem_with_space
        print('calculation\n', s_token)
        s_token = self._tokenizer.tokenize(s_token)
        fields['source_tokens'] = TextField(s_token, self._source_token_indexer)

        t_token = self._tokenizer.tokenize(' '.join(sample['manual_program']))
        t_token.insert(0, Token(START_SYMBOL))
        t_token.append(Token(END_SYMBOL))
        fields['target_tokens'] = TextField(t_token, self._target_token_indexer)
        fields['source_nums'] = MetadataField(sample['numbers'])
        fields['choice_nums'] = MetadataField(sample['choice_nums'])
        fields['label'] = MetadataField(sample['label'])

        fields['problem_form'] = MetadataField(None)
        type = self.subset_dict[sample['id']]
        fields['problem_type'] = MetadataField(type)

        return Instance(fields)

    def proving_to_instance(self, sample) -> Instance:
        fields = {}

        image = sample['img']
        image = process_image(image)
        image = image/255
        image = image.transpose(2, 0, 1)
        fields['image'] = ArrayField(image)

        # add elements to input
        sample['question'] += ' (Elements: '
        for element in sample['elements']:
            sample['question'] = sample['question'] + ' ' + element + ' , '
        sample['question'] += ')'

        question = self.process_proving_English_input(sample['question'])

        # add prefix
        question = 'Proving: ' + question
        print('proving\n', question)

        s_token = self._tokenizer.tokenize(question)
        fields['source_tokens'] = TextField(s_token, self._source_token_indexer)

        t_token = self._tokenizer.tokenize(' '.join(sample['proving_sequence']))
        t_token.insert(0, Token(START_SYMBOL))
        t_token.append(Token(END_SYMBOL))
        fields['target_tokens'] = TextField(t_token, self._target_token_indexer)

        fields['source_nums'] = MetadataField(None)
        # fields['answer'] = MetadataField(None)

        fields['problem_form'] = MetadataField(sample['problem_form'])
        fields['problem_type'] = MetadataField(sample['problem_type'])
        # fields['reasoning_skill'] = MetadataField(sample['reasoning_skill'])

        fields['source_nums'] = MetadataField(None)
        fields['choice_nums'] = MetadataField(None)
        fields['label'] = MetadataField(None)

        return Instance(fields)


