import os
import json
import glob

import tqdm


outputs = []


for question_file in tqdm.tqdm(glob.glob('RACE/train/**/*.txt')):
    outputs.append(json.load(open(question_file, 'r')))

with open('train.json', 'w') as output_file:
    json.dump(outputs, output_file)
