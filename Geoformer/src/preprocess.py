import torch
from copy import deepcopy


def corrupt_bart(input_text, mask_ratio=0.30, prefix="denoise text:"):
    """BART-style Masked Language Modeling with corrupted span prediction
    Args:
        text

    Returns:
        source_text (masked_text)
        target_text

    Ex) (in vocab ids)
    input
        In this tutorial, we’ll explore how to preprocess your data using Transformers. The main tool for this is what we call a tokenizer.

    masked_text
        denoise text: In <mask> we’ll explore how to preprocess your data <mask> Transformers. <mask> main <mask> for this is what we <mask> a tokenizer.
    target_text
        same is input text
    """

    tokens = input_text.split()

    n_tokens = len(tokens)

    n_mask = int(max(mask_ratio * n_tokens, 1))
    mask_indices = torch.randperm(n_tokens)[:n_mask].sort().values

    assert len(mask_indices) > 0, input_text

    mask_indices = mask_indices.tolist()
    span = [mask_indices[0], mask_indices[0]+1]
    spans = []

    for i, mask_index in enumerate(mask_indices):
        # if current mask is not the last one & the next mask is right after current mask
        if i < len(mask_indices) - 1 and mask_indices[i+1] == mask_index + 1:
            contiguous = True
        else:
            contiguous = False

        if contiguous:
            span[1] += 1

        else:
            # non contiguous -> output current span
            spans.append(span)
            # if current mask is not the last one -> create next span
            if i < len(mask_indices) - 1:
                span = [mask_indices[i+1], mask_indices[i+1]+1]

    masked_tokens = deepcopy(tokens)

    cum_span_length = 0
    for i, span in enumerate(spans):
        start, end = span

        masked_tokens[start-cum_span_length +
                      i: end-cum_span_length+i] = ['<mask>']

        cum_span_length += (end - start)

    masked_text = " ".join(masked_tokens)

    if prefix is None:
        source_text = masked_text
    else:
        source_text = f"{prefix} {masked_text}"

    target_text = input_text

    return source_text, target_text

