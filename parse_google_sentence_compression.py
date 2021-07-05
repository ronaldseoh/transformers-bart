import json
import csv
from glob import glob

fieldnames = ['original','compressed']

def to_csv_record(writer, buffer):
  record = json.loads(buffer)
  writer.writerow(dict(
    original=record['graph']['sentence'],
    compressed=record['compression']['text']))

with open('training_data.csv','w') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
  writer.writeheader()

  for json_file in glob('**train**.json'):
    with open(json_file) as raw_contents:
      buffer = ''
      for line in raw_contents:
        if line.strip()=='':
            to_csv_record(writer, buffer)
            buffer = ''
        else: buffer += line
      if len(buffer)>0: to_csv_record(writer, buffer)

with open('eval_data.csv','w') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=['original','compressed'])
  writer.writeheader()
  
  with open('comp-data.eval.json') as raw_contents:
    buffer = ''
    for line in raw_contents:
      if line.strip()=='':
          to_csv_record(writer, buffer)
          buffer = ''
      else: buffer += line
    if len(buffer)>0: to_csv_record(writer, buffer)
