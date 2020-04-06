#!/usr/bin/env python
# coding: utf-8

# In[1]:
import argparse
import tqdm
import spacy
from transformers import BartTokenizer
import json
from pathlib import Path
from run_bart_sum import BartSystem
import sys
sys.path.append('../../')


def process_batch(batch, model, post_processor):
    inp = [ex['question'] + ' ' + ex['option'] for ex in batch]
    inp_tensor = tokenizer.batch_encode_plus(
        inp, pad_to_max_length=True, max_length=40, return_tensors='pt')
    out_tensor = model.model.generate(
        inp_tensor["input_ids"].cuda(),
        attention_mask=inp_tensor["attention_mask"].cuda(),
        num_beams=1,
        max_length=40,
        length_penalty=1000.0,
        repetition_penalty=2.5,
        early_stopping=True,
    ).cpu()
    preds = [
        tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for g in out_tensor
    ]
    # post process

    for pred, ex in zip(preds, batch):
        ex['neural_hypothesis'] = (list(
            post_processor(pred).sents)[0]).text.strip()


def batches(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device', default=-1, type=int, help='-1 for cpu >=0 for gpu')
    parser.add_argument('--data_folder', type=Path, required=True)
    parser.add_argument('--model_checkpoint', type=Path, required=True)
    parser.add_argument('--set', default='dev')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    device_ = args.device

    if device_ > -1:
        device = f'cuda:{device_}'
    else:
        device = 'cpu'

    model = BartSystem.load_from_checkpoint(
        str(args.model_checkpoint.absolute()), map_location='cuda:0')
    model.model.to(device)

    tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')

    spacy_nlp = spacy.load("en_core_web_sm")

    # read input data
    set_ = args.set
    input_file = (args.data_folder / set_).with_suffix('.json')

    with open(input_file) as f:
        samples = json.load(f)

    batch_size = 128

    for batch in tqdm.tqdm(
            batches(samples, batch_size),
            total=len(samples) // batch_size + 1):
        process_batch(batch, model, spacy_nlp)

# write output
    output_file = input_file.absolute().parent / (f'{set_}' + '_neural.json')

    if output_file.is_file():
        for i in range(1, 100):
            output_file = input_file.absolute().parent / (
                f'{set_}' + '_neural', str(i), '.json')

            if output_file.is_file():
                continue

    print(f'Writing to {output_file}')

    with open(output_file, 'w') as f:
        json.dump(samples, f)

    sublist = samples[10:20]
    print('---Question+option')

    for s in sublist:
        print(f"{s['question'] +' '+s['option']}")
    print('---Rule based---')

    for s in sublist:
        print(f"{s['hypothesis']}")
    print('---Neural (BART) based---')

    for s in sublist:
        print(f"{s['neural_hypothesis']}")
