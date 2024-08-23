from models import LUARSimilar, SBERT, STEL, RoBERTa, MixFeaturesConfig, MixFeaturesContrastiveModel
import argparse
from datasets import load_dataset
from transformers import (
    set_seed, 
    AutoTokenizer, 
)
import torch
from tqdm import tqdm
import json
import os

def infer_RoBERTa(pretrained_path, output_path, test_dataset):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    def preprocess_function(sample):
        inputs1 = tokenizer(
            sample['anchor_utterances'],
            truncation=True, max_length=64, padding='max_length'
        )
        inputs2 = tokenizer(
            sample['paired_utterances'],
            truncation=True, max_length=64, padding='max_length'
        )

        return {
            'input_ids1': inputs1['input_ids'],
            'attention_mask1': inputs1['attention_mask'],
            'input_ids2': inputs2['input_ids'],
            'attention_mask2': inputs2['attention_mask'],
        }

    def data_collator(features):
        batch = {}
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.float)
        max_user1 = max([len(f['input_ids1']) for f in features])
        for feature in features:
            if len(feature['input_ids1']) < max_user1:
                feature['utter_mask1'] = [1] * len(feature['input_ids1']) + [0] * (max_user1-len(feature['input_ids1']))
                feature['input_ids1'] += [[tokenizer.pad_token_id] * 64] * (max_user1-len(feature['input_ids1']))
                feature['attention_mask1'] += [[0] * 64] * (max_user1-len(feature['attention_mask1']))
            else:
                feature['utter_mask1'] = [1] * len(feature['input_ids1'])
        
        max_user2 = max([len(f['input_ids2']) for f in features])
        for feature in features:
            if len(feature['input_ids2']) < max_user2:
                feature['utter_mask2'] = [1] * len(feature['input_ids2']) + [0] * (max_user2-len(feature['input_ids2']))
                feature['input_ids2'] += [[tokenizer.pad_token_id] * 64] * (max_user2-len(feature['input_ids2']))
                feature['attention_mask2'] += [[0] * 64] * (max_user2-len(feature['attention_mask2']))
            else:
                feature['utter_mask2'] = [1] * len(feature['input_ids2'])
        
        for k in ['input_ids1', 'attention_mask1', 'utter_mask1',
                'input_ids2', 'attention_mask2', 'utter_mask2']:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.int)
        return batch

    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    model = RoBERTa.from_pretrained(pretrained_path)

    set_seed(42)
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()

    test_dataset = load_dataset(
        'json',
        data_files=test_dataset
    )['train']
    test_dataset = test_dataset.map(preprocess_function)

    with torch.no_grad():
        with open(output_path, 'w') as f:
            for sample in tqdm(test_dataset):
                features = data_collator([sample])
                output = model(input_ids1=features['input_ids1'].to(device), 
                               attention_mask1=features['attention_mask1'].to(device),
                               utter_mask1=features['utter_mask1'].to(device), 
                               input_ids2=features['input_ids2'].to(device), 
                               attention_mask2=features['attention_mask2'].to(device), 
                               utter_mask2=features['utter_mask2'].to(device), 
                               labels=features['labels'].to(device))
                sim = output.similarity.item()
                sample.pop('input_ids1')
                sample.pop('attention_mask1')
                sample.pop('utter_mask1')
                sample.pop('input_ids2')
                sample.pop('attention_mask2')
                sample.pop('utter_mask2')
                f.write(
                    json.dumps(
                        {
                            **sample,
                            'pred_score': sim,
                        }
                    ) + '\n'
                )

def infer_SBERT(pretrained_path, output_path, test_dataset):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    def preprocess_function(sample):
        inputs1 = tokenizer(
            sample['anchor_utterances'],
            truncation=True, max_length=64, padding='max_length'
        )
        inputs2 = tokenizer(
            sample['paired_utterances'],
            truncation=True, max_length=64, padding='max_length'
        )

        return {
            'input_ids1': inputs1['input_ids'],
            'attention_mask1': inputs1['attention_mask'],
            'input_ids2': inputs2['input_ids'],
            'attention_mask2': inputs2['attention_mask'],
        }

    def data_collator(features):
        batch = {}
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.float)
        max_user1 = max([len(f['input_ids1']) for f in features])
        for feature in features:
            if len(feature['input_ids1']) < max_user1:
                feature['utter_mask1'] = [1] * len(feature['input_ids1']) + [0] * (max_user1-len(feature['input_ids1']))
                feature['input_ids1'] += [[tokenizer.pad_token_id] * 64] * (max_user1-len(feature['input_ids1']))
                feature['attention_mask1'] += [[0] * 64] * (max_user1-len(feature['attention_mask1']))
            else:
                feature['utter_mask1'] = [1] * len(feature['input_ids1'])
        
        max_user2 = max([len(f['input_ids2']) for f in features])
        for feature in features:
            if len(feature['input_ids2']) < max_user2:
                feature['utter_mask2'] = [1] * len(feature['input_ids2']) + [0] * (max_user2-len(feature['input_ids2']))
                feature['input_ids2'] += [[tokenizer.pad_token_id] * 64] * (max_user2-len(feature['input_ids2']))
                feature['attention_mask2'] += [[0] * 64] * (max_user2-len(feature['attention_mask2']))
            else:
                feature['utter_mask2'] = [1] * len(feature['input_ids2'])
        
        for k in ['input_ids1', 'attention_mask1', 'utter_mask1',
                'input_ids2', 'attention_mask2', 'utter_mask2']:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.int)
        return batch
    
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    model = SBERT.from_pretrained(pretrained_path)

    set_seed(42)
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()

    test_dataset = load_dataset(
        'json',
        data_files=test_dataset
    )['train']
    test_dataset = test_dataset.map(preprocess_function)

    with torch.no_grad():
        with open(output_path, 'w') as f:
            for sample in tqdm(test_dataset):
                features = data_collator([sample])
                output = model(input_ids1=features['input_ids1'].to(device), 
                               attention_mask1=features['attention_mask1'].to(device),
                               utter_mask1=features['utter_mask1'].to(device), 
                               input_ids2=features['input_ids2'].to(device), 
                               attention_mask2=features['attention_mask2'].to(device), 
                               utter_mask2=features['utter_mask2'].to(device), 
                               labels=features['labels'].to(device))
                sim = output.similarity.item()
                sample.pop('input_ids1')
                sample.pop('attention_mask1')
                sample.pop('utter_mask1')
                sample.pop('input_ids2')
                sample.pop('attention_mask2')
                sample.pop('utter_mask2')
                f.write(
                    json.dumps(
                        {
                            **sample,
                            'pred_score': sim,
                        }
                    ) + '\n'
                )

def infer_STEL(pretrained_path, output_path, test_dataset):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    def preprocess_function(sample):
        inputs1 = tokenizer(
            sample['anchor_utterances'],
            truncation=True, max_length=64, padding='max_length'
        )
        inputs2 = tokenizer(
            sample['paired_utterances'],
            truncation=True, max_length=64, padding='max_length'
        )

        return {
            'input_ids1': inputs1['input_ids'],
            'attention_mask1': inputs1['attention_mask'],
            'input_ids2': inputs2['input_ids'],
            'attention_mask2': inputs2['attention_mask'],
        }

    def data_collator(features):
        batch = {}
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.float)
        max_user1 = max([len(f['input_ids1']) for f in features])
        for feature in features:
            if len(feature['input_ids1']) < max_user1:
                feature['utter_mask1'] = [1] * len(feature['input_ids1']) + [0] * (max_user1-len(feature['input_ids1']))
                feature['input_ids1'] += [[tokenizer.pad_token_id] * 64] * (max_user1-len(feature['input_ids1']))
                feature['attention_mask1'] += [[0] * 64] * (max_user1-len(feature['attention_mask1']))
            else:
                feature['utter_mask1'] = [1] * len(feature['input_ids1'])
        
        max_user2 = max([len(f['input_ids2']) for f in features])
        for feature in features:
            if len(feature['input_ids2']) < max_user2:
                feature['utter_mask2'] = [1] * len(feature['input_ids2']) + [0] * (max_user2-len(feature['input_ids2']))
                feature['input_ids2'] += [[tokenizer.pad_token_id] * 64] * (max_user2-len(feature['input_ids2']))
                feature['attention_mask2'] += [[0] * 64] * (max_user2-len(feature['attention_mask2']))
            else:
                feature['utter_mask2'] = [1] * len(feature['input_ids2'])
        
        for k in ['input_ids1', 'attention_mask1', 'utter_mask1',
                'input_ids2', 'attention_mask2', 'utter_mask2']:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.int)
        return batch
    
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    model = STEL.from_pretrained(pretrained_path)

    set_seed(42)
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()

    test_dataset = load_dataset(
        'json',
        data_files=test_dataset
    )['train']
    test_dataset = test_dataset.map(preprocess_function)

    with torch.no_grad():
        with open(output_path, 'w') as f:
            for sample in tqdm(test_dataset):
                features = data_collator([sample])
                output = model(input_ids1=features['input_ids1'].to(device), 
                               attention_mask1=features['attention_mask1'].to(device),
                               utter_mask1=features['utter_mask1'].to(device), 
                               input_ids2=features['input_ids2'].to(device), 
                               attention_mask2=features['attention_mask2'].to(device), 
                               utter_mask2=features['utter_mask2'].to(device), 
                               labels=features['labels'].to(device))
                sim = output.similarity.item()
                sample.pop('input_ids1')
                sample.pop('attention_mask1')
                sample.pop('utter_mask1')
                sample.pop('input_ids2')
                sample.pop('attention_mask2')
                sample.pop('utter_mask2')
                f.write(
                    json.dumps(
                        {
                            **sample,
                            'pred_score': sim,
                        }
                    ) + '\n'
                )

def infer_LUAR(pretrained_path, output_path, test_dataset):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    def preprocess_function(sample):
        inputs1 = tokenizer(
            sample['anchor_utterances'],
            truncation=True, max_length=64, padding='max_length'
        )
        inputs2 = tokenizer(
            sample['paired_utterances'],
            truncation=True, max_length=64, padding='max_length'
        )

        return {
            'input_ids1': inputs1['input_ids'],
            'attention_mask1': inputs1['attention_mask'],
            'input_ids2': inputs2['input_ids'],
            'attention_mask2': inputs2['attention_mask'],
        }

    def data_collator(features):
        batch = {}
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.float)
        max_user1 = max([len(f['input_ids1']) for f in features])
        for feature in features:
            if len(feature['input_ids1']) < max_user1:
                feature['input_ids1'] += [[tokenizer.pad_token_id] * 64] * (max_user1-len(feature['input_ids1']))
                feature['attention_mask1'] += [[0] * 64] * (max_user1-len(feature['attention_mask1']))
        
        max_user2 = max([len(f['input_ids2']) for f in features])
        for feature in features:
            if len(feature['input_ids2']) < max_user2:
                feature['input_ids2'] += [[tokenizer.pad_token_id] * 64] * (max_user2-len(feature['input_ids2']))
                feature['attention_mask2'] += [[0] * 64] * (max_user2-len(feature['attention_mask2']))

        
        for k in ['input_ids1', 'attention_mask1',
                'input_ids2', 'attention_mask2']:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.int)
        return batch
    
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    model = LUARSimilar.from_pretrained(pretrained_path)

    set_seed(42)
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()

    test_dataset = load_dataset(
        'json',
        data_files=test_dataset
    )['train']
    test_dataset = test_dataset.map(preprocess_function)

    with torch.no_grad():
        with open(output_path, 'w') as f:
            for sample in tqdm(test_dataset):
                features = data_collator([sample])
                output = model(input_ids1=features['input_ids1'].to(device), 
                               attention_mask1=features['attention_mask1'].to(device), 
                               input_ids2=features['input_ids2'].to(device), 
                               attention_mask2=features['attention_mask2'].to(device), 
                               labels=features['labels'].to(device))
                sim = output.logits.item()
                sample.pop('input_ids1')
                sample.pop('attention_mask1')
                sample.pop('input_ids2')
                sample.pop('attention_mask2')
                f.write(
                    json.dumps(
                        {
                            **sample,
                            'pred_score': sim,
                        }
                    ) + '\n'
                )

def infer_MixFeature(RoBERTa_pretrained_path,
                     SBERT_pretrained_path,
                     STEL_pretrained_path,
                     LUAR_pretrained_path, 
                     output_path, 
                     test_dataset):
    roberta_tokenizer = AutoTokenizer.from_pretrained(RoBERTa_pretrained_path)
    sbert_tokenizer = AutoTokenizer.from_pretrained(SBERT_pretrained_path)
    stel_tokenizer = AutoTokenizer.from_pretrained(STEL_pretrained_path)
    luar_tokenizer = AutoTokenizer.from_pretrained(LUAR_pretrained_path, trust_remote_code=True)
    def preprocess_function(example):
        utters1 = []
        for u in example['anchor_utterances']:
            if len(u) == 0:
                continue
            utters1.append(u)
        utters2 = []
        for u in example['paired_utterances']:
            if len(u) == 0:
                continue
            utters2.append(u)
        luar_inputs1 = luar_tokenizer(
            utters1,
            truncation=True, 
            max_length=64, 
            padding='max_length'
        )
        sbert_inputs1 = sbert_tokenizer(
            utters1, 
            max_length=64, 
            padding='max_length', 
            truncation=True
        )
        stel_inputs1 = stel_tokenizer(
            utters1, 
            max_length=64, 
            padding='max_length', 
            truncation=True
        )
        roberta_inputs1 = roberta_tokenizer(
            utters1, 
            max_length=64, 
            padding='max_length', 
            truncation=True
        )
        luar_inputs2 = luar_tokenizer(
            utters2,
            truncation=True, 
            max_length=64, 
            padding='max_length'
        )
        sbert_inputs2 = sbert_tokenizer(
            utters2, 
            max_length=64, 
            padding='max_length', 
            truncation=True
        )
        stel_inputs2 = stel_tokenizer(
            utters2, 
            max_length=64, 
            padding='max_length', 
            truncation=True
        )
        roberta_inputs2 = roberta_tokenizer(
            utters2, 
            max_length=64, 
            padding='max_length', 
            truncation=True
        )
        return {
            'luar_input_ids1': luar_inputs1['input_ids'],
            'luar_attention_mask1': luar_inputs1['attention_mask'],
            'luar_input_ids2': luar_inputs2['input_ids'],
            'luar_attention_mask2': luar_inputs2['attention_mask'],
            'sbert_input_ids1': sbert_inputs1['input_ids'],
            'sbert_attention_mask1': sbert_inputs1['attention_mask'],
            'sbert_input_ids2': sbert_inputs2['input_ids'],
            'sbert_attention_mask2': sbert_inputs2['attention_mask'],
            'stel_input_ids1': stel_inputs1['input_ids'],
            'stel_attention_mask1': stel_inputs1['attention_mask'],
            'stel_input_ids2': stel_inputs2['input_ids'],
            'stel_attention_mask2': stel_inputs2['attention_mask'],
            'roberta_input_ids1': roberta_inputs1['input_ids'],
            'roberta_attention_mask1': roberta_inputs1['attention_mask'],
            'roberta_input_ids2': roberta_inputs2['input_ids'],
            'roberta_attention_mask2': roberta_inputs2['attention_mask'],
        }

    def data_collator(features):
        batch = {}
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.float)
        
        max_utters_1 = max([len(f['luar_input_ids1']) for f in features])
        max_utters_2 = max([len(f['luar_input_ids2']) for f in features])
        for feature in features:
            feature['utter_mask1'] = [1] * len(feature['luar_input_ids1'])
            if len(feature['luar_input_ids1']) < max_utters_1:
                feature['utter_mask1'] += [0] * (max_utters_1 - len(feature['utter_mask1']))
                feature['luar_input_ids1'] += [[luar_tokenizer.pad_token_id] * 64] * (max_utters_1-len(feature['luar_input_ids1']))
                feature['luar_attention_mask1'] += [[0] * 64] * (max_utters_1-len(feature['luar_attention_mask1']))
                feature['sbert_input_ids1'] += [[sbert_tokenizer.pad_token_id] * 64] * (max_utters_1-len(feature['sbert_input_ids1']))
                feature['sbert_attention_mask1'] += [[0] * 64] * (max_utters_1-len(feature['sbert_attention_mask1']))
                feature['stel_input_ids1'] += [[stel_tokenizer.pad_token_id] * 64] * (max_utters_1-len(feature['stel_input_ids1']))
                feature['stel_attention_mask1'] += [[0] * 64] * (max_utters_1-len(feature['stel_attention_mask1']))
                feature['roberta_input_ids1'] += [[roberta_tokenizer.pad_token_id] * 64] * (max_utters_1-len(feature['roberta_input_ids1']))
                feature['roberta_attention_mask1'] += [[0] * 64] * (max_utters_1-len(feature['roberta_attention_mask1']))

            feature['utter_mask2'] = [1] * len(feature['luar_input_ids2'])
            if len(feature['luar_input_ids2']) < max_utters_2:
                feature['utter_mask2'] += [0] * (max_utters_2 - len(feature['utter_mask2']))
                feature['luar_input_ids2'] += [[luar_tokenizer.pad_token_id] * 64] * (max_utters_2-len(feature['luar_input_ids2']))
                feature['luar_attention_mask2'] += [[0] * 64] * (max_utters_2-len(feature['luar_attention_mask2']))
                feature['sbert_input_ids2'] += [[sbert_tokenizer.pad_token_id] * 64] * (max_utters_2-len(feature['sbert_input_ids2']))
                feature['sbert_attention_mask2'] += [[0] * 64] * (max_utters_2-len(feature['sbert_attention_mask2']))
                feature['stel_input_ids2'] += [[stel_tokenizer.pad_token_id] * 64] * (max_utters_2-len(feature['stel_input_ids2']))
                feature['stel_attention_mask2'] += [[0] * 64] * (max_utters_2-len(feature['stel_attention_mask2']))    
                feature['roberta_input_ids2'] += [[roberta_tokenizer.pad_token_id] * 64] * (max_utters_2-len(feature['roberta_input_ids2']))
                feature['roberta_attention_mask2'] += [[0] * 64] * (max_utters_2-len(feature['roberta_attention_mask2']))

        for k in ['luar_input_ids1', 'luar_input_ids2', 'luar_attention_mask1', 'luar_attention_mask2',
                'sbert_input_ids1', 'sbert_input_ids2', 'sbert_attention_mask1', 'sbert_attention_mask2', 
                'stel_input_ids1', 'stel_input_ids2', 'stel_attention_mask1', 'stel_attention_mask2',
                'roberta_input_ids1', 'roberta_input_ids2', 'roberta_attention_mask1', 'roberta_attention_mask2']:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.int)
        for k in ['utter_mask1', 'utter_mask2']:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.float)
        return batch
    
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    config = MixFeaturesConfig()
    model = MixFeaturesContrastiveModel(config,
                                        luar_pretrained_path=LUAR_pretrained_path,
                                        sbert_pretrained_path=SBERT_pretrained_path,
                                        stel_pretrained_path=STEL_pretrained_path,
                                        roberta_pretrained_path=RoBERTa_pretrained_path)

    set_seed(42)
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()

    test_dataset = load_dataset(
        'json',
        data_files=args.test_dataset
    )['train']
    test_dataset = test_dataset.map(preprocess_function)

    with torch.no_grad():
        with open(args.test_result, 'w') as f:
            for sample in tqdm(test_dataset):
                features = data_collator([sample])
                output = model(luar_input_ids1=features['luar_input_ids1'].to(device),
                            luar_attention_mask1=features['luar_attention_mask1'].to(device),
                            sbert_input_ids1=features['sbert_input_ids1'].to(device),
                            sbert_attention_mask1=features['sbert_attention_mask1'].to(device),
                            stel_input_ids1=features['stel_input_ids1'].to(device),
                            stel_attention_mask1=features['stel_attention_mask1'].to(device),
                            roberta_input_ids1=features['roberta_input_ids1'].to(device),
                            roberta_attention_mask1=features['roberta_attention_mask1'].to(device),
                            utter_mask1=features['utter_mask1'].to(device),
                            luar_input_ids2=features['luar_input_ids2'].to(device),
                            luar_attention_mask2=features['luar_attention_mask2'].to(device),
                            sbert_input_ids2=features['sbert_input_ids2'].to(device),
                            sbert_attention_mask2=features['sbert_attention_mask2'].to(device),
                            stel_input_ids2=features['stel_input_ids2'].to(device),
                            stel_attention_mask2=features['stel_attention_mask2'].to(device),
                            roberta_input_ids2=features['roberta_input_ids2'].to(device),
                            roberta_attention_mask2=features['roberta_attention_mask2'].to(device),
                            utter_mask2=features['utter_mask2'].to(device),
                            labels=features['labels'].to(device))
                sim = output.similarity.item()
                sample.pop('luar_input_ids1')
                sample.pop('luar_attention_mask1')
                sample.pop('luar_input_ids2')
                sample.pop('luar_attention_mask2')
                sample.pop('sbert_input_ids1')
                sample.pop('sbert_attention_mask1')
                sample.pop('sbert_input_ids2')
                sample.pop('sbert_attention_mask2')
                sample.pop('stel_input_ids1')
                sample.pop('stel_attention_mask1')
                sample.pop('stel_input_ids2')
                sample.pop('stel_attention_mask2')
                sample.pop('roberta_input_ids1')
                sample.pop('roberta_attention_mask1')
                sample.pop('roberta_input_ids2')
                sample.pop('roberta_attention_mask2')
                sample.pop('utter_mask1')
                sample.pop('utter_mask2')
                f.write(
                    json.dumps(
                        {
                            **sample,
                            'pred_score': sim,
                        }
                    ) + '\n'
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["RoBERTa", "SBERT", "STEL", "LURA", "MixFeature"], required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--RoBERTa_pretrained_path", type=str)
    parser.add_argument("--SBERT_pretrained_path", type=str)
    parser.add_argument("--STEL_pretrained_path", type=str)
    parser.add_argument("--LUAR_pretrained_path", type=str)
    args = parser.parse_args()

    if args.model == "RoBERTa":
        if not args.pretrained_path:
            raise ValueError("RoBERTa pretrained path is required")
        infer_RoBERTa(args.pretrained_path, args.output_path, args.test_dataset)
    elif args.model == "SBERT":
        if not args.pretrained_path:
            raise ValueError("SBERT pretrained path is required")
        infer_SBERT(args.pretrained_path, args.output_path, args.test_dataset)
    elif args.model == "STEL":
        if not args.pretrained_path:
            raise ValueError("STEL pretrained path is required")
        infer_STEL(args.pretrained_path, args.output_path, args.test_dataset)
    elif args.model == "LURA":
        if not args.pretrained_path:
            raise ValueError("LUAR pretrained path is required")
        infer_LUAR(args.pre, args.output_path, args.test_dataset)
    elif args.model == "MixFeature":
        if not args.RoBERTa_pretrained_path or not args.SBERT_pretrained_path or not args.STEL_pretrained_path or not args.LUAR_pretrained_path:
            raise ValueError("RoBERTa, SBERT, STEL, and LUAR pretrained path are required")
        infer_MixFeature(args.RoBERTa_pretrained_path,
                         args.SBERT_pretrained_path,
                         args.STEL_pretrained_path,
                         args.LUAR_pretrained_path, 
                         args.output_path, 
                         args.test_dataset)
    else:
        raise ValueError("Invalid model name")


