from tqdm import tqdm
import sentencepiece as spm
import argparse
import os
import pandas as pd
import re


def make_text_file(data_file, by_line, text_file):

    os.makedirs(os.path.join(''.join(text_file.split('/')[:-1]), 'text_file'),
                exist_ok=True)

    data = pd.read_hdf(data_file)

    with open(text_file, 'w') as f:

        null_char_counter = 0
        still_null_char_counter = 0
        counter = 0

        for code_file in tqdm(data['file_content']):
            counter += 1

            if not by_line:
                _newline_regex = re.compile(r"\n")
                code_file = _newline_regex.sub(r"[EOL]", code_file)

            if '\0' in code_file:
                null_char_counter += 1
                print(f'Null char {null_char_counter}')

            null_char_regex = re.compile('\0')
            null_char_repl = null_char_regex.sub("", code_file)
            if '\0' in null_char_repl:
                still_null_char_counter += 1
                print(f'Still null char {null_char_counter}')

            tab_regex = re.compile(r'\t')
            tab_repl = tab_regex.sub("[TAB]", null_char_repl)

            f.write(tab_repl + '\n')

        print(f"num lines {counter}")


def spm_train(text_file: str, model_prefix: str, vocab_size: int,
              character_coverage: float, model_type: str, max_sentence_length):
    spm.SentencePieceTrainer.Train(input=text_file,
                                   model_prefix=model_prefix,
                                   vocab_size=vocab_size,
                                   character_coverage=character_coverage,
                                   model_type=model_type,
                                   unk_piece="[UNK]",
                                   pad_piece="[PAD]",
                                   user_defined_symbols=['[EOL]', '[TAB]'],
                                   hard_vocab_limit=False,
                                   max_sentence_length=max_sentence_length,
                                   num_threads=128,
                                   remove_extra_whitespaces=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_hdf', default=None)
    parser.add_argument('--by_line', default=False, type=bool)
    parser.add_argument('--text_file', default=None)
    parser.add_argument('--model_prefix', default=None)
    parser.add_argument('--vocab_size', default=None)
    parser.add_argument('--character_coverage', default=None)
    parser.add_argument('--model_type', default=None)
    parser.add_argument('--max_sentence_length', default=4000)

    args = parser.parse_args()
    if args.raw_hdf is not None and args.by_line is not None:
        make_text_file(args.raw_hdf, args.by_line, args.text_file)

    args = vars(args)
    del args['raw_hdf']
    del args['by_line']

    if None not in args.values():
        spm_train(**args)
