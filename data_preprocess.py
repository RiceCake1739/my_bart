import json
import os

from datasets import load_dataset
from torch.utils.data import DataLoader

save_dir = 'data'
train_file = os.path.join(save_dir, 'train.jsonl')
dev_file = os.path.join(save_dir, 'dev.jsonl')
test_file = os.path.join(save_dir, 'test.jsonl')
datasets = load_dataset("bookcorpus", split='train')
dl = DataLoader(dataset=datasets, batch_size=1, shuffle=False)
for ib, batch in enumerate(dl):
    batch['text'] = batch['text'][0]
    if 0 <= ib < 100000:
        with open(train_file, "a") as f:
            json.dump(batch, f)
            f.write('\n')
    elif 100000 <= ib < 101000:
        with open(dev_file, "a") as f:
            json.dump(batch, f)
            f.write('\n')
    elif 101000 <= ib < 102000:
        with open(test_file, "a") as f:
            json.dump(batch, f)
            f.write('\n')
    else:
        break
