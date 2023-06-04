import os

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

from my_bart import load_bookcorpus_dataset, text_infilling, shift_tokens_right

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
accelerator = Accelerator()
device = accelerator.device

config = BartConfig.from_json_file('config.json')
model = BartForConditionalGeneration(config)
unwrapped_model = accelerator.unwrap_model(model)
state_dict = torch.load('./models/bart-loss6.0819501876831055')
unwrapped_model.load_state_dict(state_dict)
model.to(device)

# model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

test_dataset = load_bookcorpus_dataset('test')
dl_test = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
all_loss = torch.tensor([]).to(device)

model, dl_test = accelerator.prepare(model, dl_test)

with torch.no_grad():
    for batch in dl_test:
        inputs = tokenizer(batch['text'], return_tensors="pt", padding=True)
        new_inputs_ids, attention_mask = text_infilling(inputs['input_ids'], tokenizer, 0.2, 3)
        new_inputs_ids = new_inputs_ids.to(device)
        attention_mask = attention_mask.to(device)
        unwrapped_model = accelerator.unwrap_model(model)
        output_ids = shift_tokens_right(inputs['input_ids'].to(device), tokenizer, unwrapped_model.config.decoder_start_token_id).to(device)
        # loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=output_ids).loss
        loss = model(input_ids=new_inputs_ids, attention_mask=attention_mask, labels=output_ids).loss
        gather_loss = accelerator.gather_for_metrics(loss)
        # if accelerator.is_local_main_process:
                #     all_loss = torch.cat((all_loss, gather_loss), dim=0)
        gather_loss = gather_loss.unsqueeze(0)
        all_loss = torch.cat((all_loss, gather_loss), dim=0)
mean_loss = all_loss.mean()
if accelerator.is_local_main_process:
    with open('./logs/log.txt', "a") as f:
        f.write("best model test loss:" + str(mean_loss))
        f.write('\n')


