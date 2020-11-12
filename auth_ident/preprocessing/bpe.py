import re


def word_tokenizer(words):
    
    split_w_space = re.split(r'(\s+)', words)
    return split_w_space
