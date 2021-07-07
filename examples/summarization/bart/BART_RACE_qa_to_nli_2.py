#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')


# In[98]:


from run_bart_sum import BartSystem
from pathlib import Path
import json
from transformers import BartTokenizer
import spacy
import tqdm
import re


# In[3]:


device_ = 0
if device_ > -1:
    device = f'cuda:{device_}'
else:
    device = 'cpu'
model = BartSystem.load_from_checkpoint('../../.models/bart9/epoch=1.ckpt', map_location='cuda:0')
model.model.to('cuda:0')


# In[4]:


tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')


# In[5]:


spacy_nlp =spacy.load("en_core_web_sm")


# In[6]:


# read input data
set_ = 'dev'
input_file = Path(f'../../.data/RACE/set3/{set_}.json')


# In[7]:


with open(input_file) as f:
    samples = json.load(f)


# In[115]:


def preprocess(q, o):
    if '_' in q: #FITB
        h = q.replace('_', o)
    else:
        h = q + ' ' + o
    return h
        


# In[140]:


dots = re.compile(r"\.([\.\'\" ]{2,}[\w ]*)")
def postprocessor(inp):
    dots_removed = dots.sub('.', inp) # leave the first group as it has single period required to end the sentence
    first_sent = (list(spacy_nlp(dots_removed).sents)[0]).text.strip()
    return first_sent


# In[141]:


def process_batch(batch, model, post_processor):
    # preprocess
    inp = [preprocess(ex['question'], ex['option']) for ex in batch]
    inp_tensor = tokenizer.batch_encode_plus(inp, pad_to_max_length=True, max_length=40, return_tensors='pt')
    out_tensor = model.model.generate(inp_tensor["input_ids"].cuda(),
            attention_mask=inp_tensor["attention_mask"].cuda(),
            num_beams=1, do_sample=False,no_repeat_ngram_size=2, top_k=2,
            max_length=40,
            early_stopping=True,).cpu()
    preds = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            for g in out_tensor
        ]
    # post process
    for pred,ex in zip(preds, batch):
        ex['neural_hypothesis'] = postprocessor(pred)


# In[142]:


batch_size = 128
def batches(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


# In[143]:


for batch in tqdm.tqdm(batches(samples, batch_size), total=len(samples)//batch_size + 1):
    process_batch(batch, model, spacy_nlp)


# In[ ]:


# write output
#output_file = input_file.absolute().parent /(f'{set_}' + '_neural.json')
#with open(output_file, 'w') as f:
#    json.dump(samples, f)


# In[146]:


sublist = samples[20:30]
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




