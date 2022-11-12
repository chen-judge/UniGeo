from transformers import T5ForConditionalGeneration, T5Tokenizer

if __name__ == '__main__':

    print('Downloading checkpoints if not cached')
    print('T5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    print('Done!')

