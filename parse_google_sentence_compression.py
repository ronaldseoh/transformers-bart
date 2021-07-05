import gzip
import json
import glob


def write_to_file(source_file, target_file, temp_buffer):

    record = json.loads(temp_buffer)

    source_file.write(record['graph']['sentence'] + '\n')
    target_file.write(record['compression']['text'] + '\n')

train_source = open('train.source','w')
train_target = open('train.target','w')

for json_file in glob.glob('**train**.json.gz'):
    with gzip.open(json_file, 'rt') as raw_contents:
        temp_buffer = ''
        for line in raw_contents:
            if line.strip() =='':
                write_to_file(train_source, train_target, temp_buffer)
                temp_buffer = ''
            else: temp_buffer += line

        if len(temp_buffer) > 0:
            write_to_file(train_source, train_target, temp_buffer)

train_source.close()
train_target.close()

dev_source = open('dev.source','w')
dev_target = open('dev.target','w')

with gzip.open('comp-data.eval.json.gz', 'rt') as raw_contents:
    temp_buffer = ''

    for line in raw_contents:
        if line.strip() == '':
            write_to_file(dev_source, dev_target, temp_buffer)
            temp_buffer = ''
        else: temp_buffer += line

    if len(temp_buffer) > 0:
        write_to_file(dev_source, dev_target, temp_buffer)

dev_source.close()
dev_target.close()
