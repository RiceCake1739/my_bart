import json
import os
import numpy as np
import math
import argparse
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, RandomSampler
import tqdm
from transformers import BartTokenizer, BartModel, Seq2SeqTrainer, BartForConditionalGeneration, TrainingArguments, \
    BartConfig
from datasets import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
accelerator = Accelerator()
device = accelerator.device


def text_infilling(input_ids, tokenizer, mask_rate, poisson_lambda):
    special_tokens_mask_inputs = np.array([
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
    ], dtype=bool)
    # 每个句子中所有可以用于mask的token的位置
    can_mask_tokens = np.array(~(input_ids == tokenizer.pad_token_id) & ~special_tokens_mask_inputs)
    # 每个句子中要mask的数量
    mask_numbers = [int(math.ceil(can_mask_token.sum() * mask_rate)) for can_mask_token in can_mask_tokens]
    # mask_numbers = [int(math.floor(can_mask_token.sum() * mask_rate)) for can_mask_token in can_mask_tokens]
    # 用泊松分布获取每个mask的长度
    mask_lengths = [np.random.poisson(lam=poisson_lambda, size=mask_number) for mask_number in mask_numbers]
    # 这里有参考span_mask_tokens的实现，尽可能截取mask总长度接近的子数组
    mask_lengths = [row[:length] for row, length in zip(mask_lengths,
                                                        [np.argmin(
                                                            np.abs(np.cumsum(mask_lengths[i], 0) - mask_numbers[i])) + 1
                                                         for i in range(input_ids.shape[0])])]
    # 如果有mask=0，句子后要新增的padding长度
    padding_length = max(np.sum(mask_length == 0) for mask_length in mask_lengths)
    indices = [np.where(can_mask_token == 1) for can_mask_token in can_mask_tokens]
    # indices = np.argwhere(can_mask_tokens == 1)
    # 挑选用于mask的token下标
    start_indexs = list([np.sort(np.random.choice(indice[0], mask_lengths[i].shape[0], replace=False)) for i, indice in
                         enumerate(indices)])
    # 初始化经过masked后的输入
    masked_inputs_id = torch.tensor(np.full((input_ids.shape[0], input_ids.shape[1] + padding_length),
                                            fill_value=tokenizer.pad_token_id))
    for i, masked_input_id in enumerate(masked_inputs_id):
        raw_index = 0
        cur_mask_length_index = 0
        for j in range(len(masked_input_id)):
            if raw_index >= input_ids.shape[1]:
                break
            # 到需要mask的下标时
            if cur_mask_length_index < len(start_indexs[i]) and raw_index >= start_indexs[i][cur_mask_length_index]:
                masked_input_id[j] = tokenizer.mask_token_id
                raw_index = start_indexs[i][cur_mask_length_index] + mask_lengths[i][cur_mask_length_index]
                cur_mask_length_index += 1
                if cur_mask_length_index == len(start_indexs[i]) and \
                        (raw_index >= input_ids.shape[1] or input_ids[i][raw_index] == tokenizer.pad_token_id):
                    masked_input_id[j + 1] = tokenizer.eos_token_id
                    break
            # 没到需要mask的下标时
            else:
                masked_input_id[j] = input_ids[i][raw_index]
                raw_index += 1
    # 生成经过text_infilling后的输入的attention_mask
    new_attention_mask = np.where(masked_inputs_id == tokenizer.pad_token_id, 0, 1)
    new_attention_mask = torch.tensor(new_attention_mask)

    return masked_inputs_id, new_attention_mask


# text_infilling implementation code in transformers
def span_mask_tokens(input_ids, tokenizer, mask_rate):
    """
    Sampling text spans with span lengths drawn from a Poisson distribution and masking them.
    """
    special_tokens_mask_inputs = np.array([
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
    ], dtype=bool)
    # determine how many tokens we need to mask in total
    is_token_mask = np.array(~(input_ids == tokenizer.pad_token_id) & ~special_tokens_mask_inputs)
    num_tokens_to_mask = int(math.ceil(is_token_mask.sum() * mask_rate))  # mask_rate
    if num_tokens_to_mask == 0:
        return input_ids
    # generate a sufficient number of span lengths
    span_lengths = np.random.poisson(lam=3, size=(num_tokens_to_mask,))  # poisson center
    while np.cumsum(span_lengths, 0)[-1] < num_tokens_to_mask:
        span_lengths = np.concatenate(
            [span_lengths, np.random.poisson(lam=3, size=(num_tokens_to_mask,))]
        )
    # remove all spans of length 0
    # note that BART inserts additional mask tokens where length == 0,
    # which we do not implement for now as it adds additional complexity
    # 去除mask=0的情况
    span_lengths = span_lengths[span_lengths > 0]
    # trim to about num_tokens_to_mask tokens
    cutoff_idx = np.argmin(np.abs(np.cumsum(span_lengths, 0) - num_tokens_to_mask)) + 1
    span_lengths = span_lengths[:cutoff_idx]

    # randomly choose starting positions for masking
    token_indices = np.argwhere(is_token_mask == 1)
    # 是在一个batch中所有可选的indices中随机选，这样可能导致某些input被mask特别多次，某些input几乎没有被mask
    span_starts = np.random.permutation(token_indices.shape[0])[: span_lengths.shape[0]]
    # prepare mask
    masked_indices = np.array(token_indices[span_starts])
    mask = np.full_like(input_ids, dtype=bool, fill_value=False)
    # mask starting positions
    for mi in masked_indices:
        mask[tuple(mi)] = True
    span_lengths -= 1
    # fill up spans
    max_index = input_ids.shape[1] - 1
    remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)
    while np.any(remaining):
        masked_indices[remaining, 1] += 1
        for mi in masked_indices:
            mask[tuple(mi)] = True
        span_lengths -= 1
        remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)
    # place the mask tokens
    mask[np.where(special_tokens_mask_inputs)] = False
    input_ids[np.where(mask)] = tokenizer.mask_token_id

    # remove mask tokens that are not starts of spans
    to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
    new_input_ids = np.full_like(input_ids, fill_value=tokenizer.pad_token_id)
    for i, example in enumerate(input_ids):
        new_example = example[~to_remove[i]]
        new_input_ids[i, : new_example.shape[0]] = new_example

    return new_input_ids, input_ids


def shift_tokens_right(input_ids, tokenizer, decoder_start_token_id):
    # shifted_input_ids = torch.tensor(np.full((input_ids.shape[0], input_ids.shape[1] + 1), fill_value=0)).to(device)
    # shifted_input_ids[:, 1:] = input_ids
    # shifted_input_ids[:, 0] = decoder_start_token_id
    # shifted_input_ids = torch.where(shifted_input_ids == tokenizer.pad_token_id, torch.tensor(-100).to(device), shifted_input_ids)
    # shifted_input_ids = torch.where(input_ids == tokenizer.pad_token_id, torch.tensor(-100).to(device), input_ids)
    return input_ids


def load_bookcorpus_dataset(split_name):
    load_dir = 'data'
    dataset = []
    dev_dataset_path = os.path.join(load_dir, '{}.jsonl'.format(split_name))
    with open(dev_dataset_path) as f:
        for line in f:
            dataset.append({'text': json.loads(line)['text']})
    return dataset


def train(num_epochs, learning_rate, mask_rate, poisson_lambda):
    # config = BartConfig.from_json_file('config.json')
    # model = BartForConditionalGeneration(config)
    # model.to(device)
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_dataset = load_bookcorpus_dataset('train')
    dl_train = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    dev_dataset = load_bookcorpus_dataset('dev')
    dl_dev = DataLoader(dataset=dev_dataset, batch_size=32, shuffle=False)
    # test_dataset = load_bookcorpus_dataset('test')
    # dl_test = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    model, optimizer, dl_train = accelerator.prepare(model, optimizer, dl_train)
    dl_dev = accelerator.prepare(dl_dev)
    # dl_test = accelerator.prepare(dl_test)

    best_val = None
    not_update_epoch = 0
    for epi in range(num_epochs):
        ite = enumerate(dl_train)
        ite = tqdm.tqdm(ite, total=len(dl_train), disable=not accelerator.is_local_main_process)
        for ib, batch in ite:
            inputs = tokenizer(batch['text'], return_tensors="pt", padding=True)
            new_inputs_ids, attention_mask = text_infilling(inputs['input_ids'], tokenizer, mask_rate, poisson_lambda)
            new_inputs_ids = new_inputs_ids.to(device)
            attention_mask = attention_mask.to(device)
            unwrapped_model = accelerator.unwrap_model(model)
            output_ids = shift_tokens_right(inputs['input_ids'].to(device), tokenizer,
                                            unwrapped_model.config.decoder_start_token_id).to(device)
            loss = model(input_ids=new_inputs_ids, attention_mask=attention_mask, labels=output_ids).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            all_loss = torch.tensor([]).to(device)
            for batch in dl_dev:
                inputs = tokenizer(batch['text'], return_tensors="pt", padding=True).to(device)
                unwrapped_model = accelerator.unwrap_model(model)
                output_ids = shift_tokens_right(inputs['input_ids'], tokenizer,
                                                unwrapped_model.config.decoder_start_token_id).to(
                    device)
                loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                             labels=output_ids).loss
                gather_loss = accelerator.gather_for_metrics(loss)
                all_loss = torch.cat((all_loss, gather_loss), dim=0)
                # inputs = tokenizer(batch['text'], return_tensors="pt", padding=True)
                # new_inputs_ids, attention_mask = text_infilling(inputs['input_ids'], tokenizer, mask_rate,
                #                                                 poisson_lambda)
                # new_inputs_ids = new_inputs_ids.to(device)
                # attention_mask = attention_mask.to(device)
                # unwrapped_model = accelerator.unwrap_model(model)
                # output_ids = shift_tokens_right(inputs['input_ids'], tokenizer, unwrapped_model.config.decoder_start_token_id).to(
                #     device)
                # loss = model(input_ids=new_inputs_ids, attention_mask=attention_mask, labels=output_ids).loss
            mean_loss = all_loss.mean()
            accelerator.print(mean_loss)
            if accelerator.is_local_main_process:
                with open('./logs/log.txt', "a") as f:
                    f.write('epoch ' + str(epi) + ' loss:' + str(mean_loss))
                    f.write('\n')
            if best_val is None or mean_loss < best_val:
                best_val = mean_loss
                not_update_epoch = 0
            else:
                not_update_epoch += 1
            if not_update_epoch >= 10:
                return
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), './models/bart-loss{}'.format(best_val))

            # unwrapped_model = accelerator.unwrap_model(model)
            # outputs = unwrapped_model.generate(new_inputs_ids, attention_mask=attention_mask, max_length=128)
            # output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # inputs = tokenizer(["usually , he would be tearing around the living room , playing with his toys .",
    #                     "instead , his baby blues remained focused on the television ."],
    #                    return_tensors="pt",
    #                    padding=True)['input_ids']
    # print(inputs)
    # output_ids = shift_tokens_right(inputs, tokenizer, model.config.decoder_start_token_id)
    # print(output_ids)
    # # new_inputs, attention_mask = text_infilling(inputs['input_ids'], tokenizer, mask_rate, poisson_lambda)
    # print(new_inputs)
    # print(attention_mask)
    # print(tokenizer.decode(new_inputs[0]))
    # print(tokenizer.decode(new_inputs[1]))
    # new_inputs = span_mask_tokens(inputs['input_ids'], tokenizer, mask_rate)
    # print(tokenizer.decode(new_inputs[0]))
    # print(tokenizer.decode(new_inputs[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_rate", type=float, default=0.05, help="Rate of tokens to be masked")
    parser.add_argument("--poisson_lambda", type=int, default=3, help="Poisson distribution lambda")
    parser.add_argument("--learning_rate", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Num of epochs")
    args = parser.parse_args()
    train(**args.__dict__)
