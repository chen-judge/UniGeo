import re
import cv2 as cv
import numpy as np


def process_image(img, min_side=224):
    # fill the diagram with a white background and resize it
    size = img.shape
    h, w = size[0], size[1]

    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv.resize(img, (new_w, new_h))

    top, bottom, left, right = 0, min_side-new_h, 0, min_side-new_w
    pad_img = cv.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right),
                                cv.BORDER_CONSTANT, value=[255,255,255])
    pad_img = pad_img / 255

    return pad_img


def create_patch(patch_num=7):
    bboxes = []
    for i in range(patch_num):
        for j in range(patch_num):
            box = [1.0 * j / patch_num, 1.0 * i / patch_num, 1.0 * (j + 1) / patch_num, 1.0 * (i + 1) / patch_num]
            bboxes.append(box)
    bboxes = np.array(bboxes)
    return bboxes


def split_elements(sequence):
    new_sequence = []
    for token in sequence:
        if 'N_' in token or 'NS_' in token or 'frac' in token:
            new_sequence.append(token)
        elif token.istitle():
            new_sequence.append(token)
        elif re.search(r'[A-Z]', token):
            # split geometry elements with a space: ABC -> A B C
            new_sequence.extend(token)
        else:
            new_sequence.append(token)

    return new_sequence


def process_english_text(ori_text):
    text = re.split(r'([=≠≈+-/△∠∥⊙☉⊥⟂≌≅▱∽⁀⌒;,:.•?])', ori_text)
    text = ' '.join(text)

    text = text.split()
    text = split_elements(text)
    text = ' '.join(text)

    # The initial version of the calculation problem (GeoQA) is in Chinese.
    # The translated English version still contains some Chinese tokens,
    # which should be replaced by English words.
    replace_dict ={'≠': 'not-equal', '≈': 'approximate', '△': 'triangle', '∠': 'angle', '∥': 'parallel',
                   '⊙': 'circle', '☉': 'circle', '⊥': 'perpendicular', '⟂': 'perpendicular', '≌': 'congruent', '≅': 'congruent',
                   '▱': 'parallelogram', '∽': 'similar', '⁀': 'arc', '⌒': 'arc'
                   }
    for k, v in replace_dict.items():
        text = text.replace(k, v)

    return text


def process_Chinese_solving(ori_text):
    index = ori_text.find('故选')
    text = ori_text[:index]

    # delete special tokens
    delete_list = ['^{°}', '{', '}', '°', 'cm', 'm', '米', ',', ':', '．', '、', '′', '~', '″', '【', '】', '$']
    for d in delete_list:
        text = text.replace(d, ' ')
    # delete Chinese tokens
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    text1 = re.sub(zh_pattern, ' ', text)

    # split
    pattern = re.compile(r'([=≠≈+-]|π|×|\\frac|\\sqrt|\\cdot|\√|[∵∴△∠∥⊙☉⊥⟂≌≅▱∽⁀⌒]|[.,，：；;,:.•?]|\d+\.?\d*|)')
    text2 = re.split(pattern, text1)
    # split elements: ABC -> A B C
    text2 = split_elements(text2)

    # store numbers
    text3 = []
    nums = []
    # replace only nums
    for t in text2:
        if re.search(r'\d', t):  # NS: number in solving
            if float(t) in nums:
                text3.append('NS_'+str(nums.index(float(t))))
            else:
                text3.append('NS_'+str(len(nums)))
                nums.append(float(t))
        else:
            text3.append(t)

    # replace
    text4 = ' '.join(text3)
    replace_dict = {'≠': 'not-equal', '≈': 'approximate', '△': 'triangle', '∠': 'angle', '∥': 'parallel',
                    '⊙': 'circle', '☉': 'circle', '⊥': 'perpendicular', '⟂': 'perpendicular', '≌': 'congruent', '≅': 'congruent',
                    '▱': 'parallelogram', '∽': 'similar', '⁀': 'arc', '⌒': 'arc',
                    '/ /': 'parallel', '∵': 'because', '∴': 'therefore', '²': 'square', '√': 'root'
                    }
    for k, v in replace_dict.items():
        text4 = text4.replace(k, v)
    text4 = text4.split()

    return text4, nums