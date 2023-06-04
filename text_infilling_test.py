from transformers import BartTokenizer

from my_bart import text_infilling

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

inputs = tokenizer(["usually , he would be tearing around the living room , playing with his toys .",
                    "instead , his baby blues remained focused on the television ."],
                   return_tensors="pt",
                   padding=True)['input_ids']
new_inputs, _ = text_infilling(inputs, tokenizer, 0.1, 3)
print(inputs)
print(tokenizer.batch_decode(inputs))
print(new_inputs)
print(tokenizer.batch_decode(new_inputs))
