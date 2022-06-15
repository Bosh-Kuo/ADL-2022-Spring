import argparse
import json
import math
import numpy as np
import os
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import get_scheduler
from transformers import T5ForConditionalGeneration

def train(model, tokenizer, device, train_loader, valid_loader, gradient_accumulation_steps, optimizer, scheduler, epoch, output_dir, gen_kwargs):
	# initialize the parameters
	best_loss = float('inf')

	for ep in range(epoch): 
		# set the model as training mode
		model.train()

		# set tdqm loop
		loop = tqdm(enumerate(train_loader), total = len(train_loader))

		# initialize parameters
		train_loss = 0
		
		for i, data in loop:
			# push input_ids, attention_mask to the device
			data = data.to(device)

			# put the seq into model and get the output token and then turn into numpy
			output = model(**data)
			
			# get the probability of each token
			# output logits: (batch, length, vocab_size)
			prob = F.softmax(output['logits'], dim = 2)
			
			# get the maximum probability of each token
			prob = prob.max(2, keepdim = True)[0]

			# turn the probability into log form
			log_prob = torch.log(prob).sum()

			# put the seq into model and get the output token and then turn into numpy
			output_sample = model.generate(data['input_ids'], attention_mask = data['attention_mask'], **gen_kwargs)

			# Replace -100 in the labels as we can't decode them.
			labels = data['labels'].cpu().numpy()
			labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

			decoded_preds_sample = tokenizer.batch_decode(output_sample, skip_special_tokens = True)

			# use hit rate as reward function
			reward = hit(decoded_preds_sample)
			
			# calculate loss (policy based RL)
			rl_loss = -reward * log_prob

			# since we perform gradient ascent, get the negative of loss
			loss = rl_loss * 0.5 + output.loss * 0.5
			loss = loss / gradient_accumulation_steps
			loss.backward()

			loop.set_description(f'Epoch: [{ep+1}/{epoch}]')
			loop.set_postfix(loss = loss.item(), lr = optimizer.param_groups[0]['lr'])

			if i % gradient_accumulation_steps == 0 or i == len(train_loader) - 1:
				# update the model and train_loss
				optimizer.step()
				scheduler.step()
				
				# set graident of optimizer to zero
				optimizer.zero_grad()

		test_loss = test(model, tokenizer, device, valid_loader, gen_kwargs)
		print(f'Testing loss: {test_loss}')

		if test_loss < best_loss:
			best_loss = test_loss
			model.save_pretrained(output_dir)		
	
def test(model, tokenizer, device, valid_loader, gen_kwargs):
	# set the model as eval mode
	model.eval()

	# parameters
	test_loss = 0

	# we don't perform back-propagation now, free the GPU memory
	with torch.no_grad():
		# set tdqm loop
		loop = tqdm(enumerate(valid_loader), total = len(valid_loader))

		for i, data in loop:
			# push input_ids, attention_mask to the device
			data = data.to(device)

			# put the seq into model and get the output token and then turn into numpy
			output = model(**data)

			test_loss += output.loss

			loop.set_description(f'Validation ')

	return test_loss / len(valid_loader.dataset)			

def preprocess_function(examples):
	inputs = [ex for ex in examples['source']]
	targets = [ex for ex in examples['target']]
	model_inputs = tokenizer(
		inputs, max_length=args.max_input_length, truncation=True, padding='max_length',
		add_special_tokens=True,
	)

	# Set up the tokenizer for targets
	with tokenizer.as_target_tokenizer():
		labels = tokenizer(
			targets, max_length=args.max_target_length, truncation=True, padding='max_length',
			add_special_tokens=True,
		)

	model_inputs["labels"] = labels["input_ids"]
	return model_inputs
	
def read_data(data_dir):
	splits = ['train', 'validation', 'test']
	print(data_dir)
	datasets = {}
	for split in splits:
		directory = os.path.join(data_dir, split)
		datasets[split] = load_dataset('json', data_files=f'{directory}/hit_generator_text.json', field='data')  
		if split != 'test':
			datasets[split] = datasets[split].map(
				preprocess_function,
				batched=True,
				remove_columns=['source', 'target', 'intent'],
			)['train']
		else:
			datasets[split] = datasets[split]['train']
	return datasets['train'], datasets['validation'], datasets['test']

def hit(sentences):
	hit_num = 0
	# start with the second utterance from the simulator
	for sentence in sentences:
		lemma_utterance = [token.lemma_ for token in nlp(sentence)]
		service_hits = defaultdict(int)
		for key, (one, multi) in keywords.items():
			intersection = set(one) & set(lemma_utterance)
			# check whether the word, the length is bigger than 2, is in the utterance
			for m in multi:
				unsplit_utterance = " ".join(lemma_utterance)
				if m in unsplit_utterance:
					intersection.add(m)
			service_hits[key] += len(intersection)
		# Is there a keyword in this utterance
		isService = sum(service_hits.values()) != 0

		if isService:
			hit_num += 1

	return hit_num / len(sentences)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# dataset args
	parser.add_argument('--data_root', default = './data')
	parser.add_argument('--keywords_file', default = '../final_project_scripts/keywords.json')
	
	# model args
	parser.add_argument('--model_name_or_path', default = './t5_hit_generator')
	parser.add_argument('--max_input_length', type = int, default = 128)
	parser.add_argument('--max_target_length', type = int, default = 32)
	parser.add_argument('--do_sample', type = bool, default = True)
	parser.add_argument('--top_k',type = int, default = 80)
	parser.add_argument('--top_p',type = float, default = 0.95)
	parser.add_argument('--repetition_penalty',type = float, default = 0.7)
	parser.add_argument('--temperature', type = float, default = 0.65)
	parser.add_argument('--seed', type = int, default = 521)
	parser.add_argument('--output_dir',default = './RL_finetune')

	# training args
	parser.add_argument('--batch_size', type = int, default = 16)
	parser.add_argument('--gradient_accumulation_steps', type = int, default = 1)
	parser.add_argument('--lr', type = float, default = 5e-5)
	parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
	parser.add_argument('--lr_scheduler_type', default = 'linear')
	parser.add_argument('--epoch', type = int, default = 5)
	parser.add_argument('--plot', type = bool, default = True)
	args = parser.parse_args()

	# read keywords
	with open(os.path.join(args.keywords_file)) as f:
		keywords = json.load(f)

	nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
	# lemmatize words in keywords
	for key, val in keywords.items():
		# separate words by its length (one, others)
		one_lemma = []
		multi_lemma = []
		for word in val:
			split = [token.lemma_ for token in nlp(word)]
			if len(split) >= 2:
				multi_lemma.append(" ".join(split))
			else:
				one_lemma.append(split[0])
			keywords[key] = [one_lemma, multi_lemma]

	# make the model and tokenizer dir if it doesn't exist
	if not os.path.isdir(os.path.join(args.output_dir)):
		os.mkdir(os.path.join(args.output_dir))
	
	# load model and tokenizer
	model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
	tokenizer.pad_token = tokenizer.eos_token

	# read dataset
	train_dataset, eval_dataset, test_dataset = read_data(args.data_root)
	print(train_dataset)

	# define data collator
	data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

	# dataloader
	train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
	valid_loader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size)

	data_iter = iter(train_loader)
	data = data_iter.next()
	print('data size in each batch: ' ,data['input_ids'].shape)

	# check if GPU is available
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = torch.device(device)
	print('Device: ', device)

	# push model to device
	model.to(device)
	#print(model)

	# set seed
	torch.manual_seed(args.seed)

	# optimizer
	# Split weights in two groups, one with weight decay and the other not.
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{
			"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
			"weight_decay": 0.0,
		},
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

	# scheduler and math around the number of training steps.
	num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
	scheduler = get_scheduler(
		name=args.lr_scheduler_type,
		optimizer=optimizer,
		num_warmup_steps=0,
		num_training_steps= args.epoch * num_update_steps_per_epoch,
	)

	# settings for generating titles
	gen_kwargs = {
			"max_length": args.max_target_length,
			"do_sample": args.do_sample,
			"top_k": args.top_k,
			"top_p": args.top_p,
			"temperature": args.temperature,
			"repetition_penalty": args.repetition_penalty
		}


	# train
	train(model, tokenizer, device, train_loader, valid_loader, args.gradient_accumulation_steps, optimizer, scheduler, args.epoch, args.output_dir, gen_kwargs)

	# save tokenizer
	tokenizer.save_pretrained(os.path.join(args.output_dir))