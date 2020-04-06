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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device', default=-1, type=int, help='-1 for cpu >=0 for gpu')
    parser.add_argument('--data_folder', type=Path, required=True)


device_ = 0

if device_ > -1:
    device = f'cuda:{device_}'
else:
    device = 'cpu'
model = BartSystem.load_from_checkpoint(
    '../../.models/bart2/epoch=1_v0.ckpt', map_location='cuda:0')
model.model.to('cuda:0')

# In[4]:

tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')

# In[5]:

spacy_nlp = spacy.load("en_core_web_sm")

# In[6]:

# read input data
set_ = 'dev'
input_file = Path(f'../../.data/RACE/set3/{set_}.json')

# In[7]:

with open(input_file) as f:
    samples = json.load(f)

# In[8]:


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


# In[9]:

batch_size = 128


def batches(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


# In[10]:

for batch in tqdm.tqdm(
        batches(samples, batch_size), total=len(samples) // batch_size + 1):
    process_batch(batch, model, spacy_nlp)

# In[11]:

# write output
output_file = input_file.absolute().parent / (f'{set_}' + '_neural.json')

if output_file.is_file():
    for i in range(1, 100):
        output_file = input_file.absolute().parent / (f'{set_}' + '_neural',
                                                      str(i), '.json')

        if output_file.is_file():
            continue

print(f'Writing to {output_file}')

with open(output_file, 'w') as f:
    json.dump(samples, f)

# In[17]:

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

# In[ ]:
