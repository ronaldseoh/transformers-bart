import pandas as pd


# Load train.tsv and dev.tsv from qa2d
train = pd.read_table('train.tsv')

train_source = open('train.source', 'w')
train_target = open('train.target', 'w')

for n in range(len(train)):
    train_source.write(
        str(train['question'].iloc[n]) + ' ' + str(train['answer'].iloc[n]) + '\n')

    train_target.write(train['turker_answer'].iloc[n] + '\n')

train_source.close()
train_target.close()

dev = pd.read_table('dev.tsv')

dev_source = open('dev.source', 'w')
dev_target = open('dev.target', 'w')

for n in range(len(dev)):
    dev_source.write(
        str(dev['question'].iloc[n]) + ' ' + str(dev['answer'].iloc[n]) + '\n')

    dev_target.write(dev['turker_answer'].iloc[n] + '\n')

dev_source.close()
dev_target.close()
