from models import LUARSimilar, SBERT, STEL, RoBERTa
import argparse
from datasets import load_dataset
from transformers import (
    RoBERTa,
    set_seed, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback,
)
import torch
import numpy as np
import evaluate


def fine_tune_RoBERTa(pretrained_path, model_save_path, train_dataset, dev_dataset):
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
    
    model = RoBERTa.from_pretrained(pretrained_path)

    set_seed(42)

    train_dataset = load_dataset(
        'json',
        data_files=train_dataset,
    )['train']
    train_dataset = train_dataset.shuffle().map(preprocess_function, num_proc=100)
    eval_dataset = load_dataset(
        'json',
        data_files=dev_dataset,
    )['train']
    eval_dataset = eval_dataset.shuffle().map(preprocess_function, num_proc=100)

    metric = evaluate.load("roc_auc")
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds)
        result = metric.compute(prediction_scores=preds, references=p.label_ids)
        return result

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=model_save_path,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=256,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            num_train_epochs=3,
            evaluation_strategy='steps',
            eval_steps=0.1,
            save_strategy='steps',
            save_steps=0.1,
            logging_strategy='steps',
            logging_steps=5,
            label_names=['labels'],
            report_to='tensorboard',
            save_total_limit=3,
            load_best_model_at_end=True,
            remove_unused_columns=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()
    trainer.save_model(model_save_path)

def fine_tune_SBERT(pretrained_path, model_save_path, train_dataset, dev_dataset):
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
    
    model = SBERT.from_pretrained(pretrained_path)

    set_seed(42)

    train_dataset = load_dataset(
        'json',
        data_files=train_dataset,
    )['train']
    train_dataset = train_dataset.shuffle().map(preprocess_function, num_proc=100)
    eval_dataset = load_dataset(
        'json',
        data_files=dev_dataset,
    )['train']
    eval_dataset = eval_dataset.shuffle().map(preprocess_function, num_proc=100)

    metric = evaluate.load("roc_auc")
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds)
        result = metric.compute(prediction_scores=preds, references=p.label_ids)
        return result

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=model_save_path,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=256,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            num_train_epochs=3,
            evaluation_strategy='steps',
            eval_steps=0.1,
            save_strategy='steps',
            save_steps=0.1,
            logging_strategy='steps',
            logging_steps=5,
            label_names=['labels'],
            report_to='tensorboard',
            save_total_limit=3,
            load_best_model_at_end=True,
            remove_unused_columns=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()
    trainer.save_model(model_save_path)

def fine_tune_STEL(pretrained_path, model_save_path, train_dataset, dev_dataset):
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
    
    model = STEL.from_pretrained(pretrained_path)

    set_seed(42)

    train_dataset = load_dataset(
        'json',
        data_files=train_dataset,
    )['train']
    train_dataset = train_dataset.shuffle().map(preprocess_function, num_proc=100)
    eval_dataset = load_dataset(
        'json',
        data_files=dev_dataset,
    )['train']
    eval_dataset = eval_dataset.shuffle().map(preprocess_function, num_proc=100)

    metric = evaluate.load("roc_auc")
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds)
        result = metric.compute(prediction_scores=preds, references=p.label_ids)
        return result

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=model_save_path,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=256,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            num_train_epochs=3,
            evaluation_strategy='steps',
            eval_steps=0.1,
            save_strategy='steps',
            save_steps=0.1,
            logging_strategy='steps',
            logging_steps=5,
            label_names=['labels'],
            report_to='tensorboard',
            save_total_limit=3,
            load_best_model_at_end=True,
            remove_unused_columns=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    train_result = trainer.train()
    trainer.save_model(model_save_path)

def fine_tune_LURA(pretrained_path, model_save_path, train_dataset, dev_dataset):
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

    model = LUARSimilar.from_pretrained(pretrained_path)

    set_seed(42)

    train_dataset = load_dataset(
        'json',
        data_files=train_dataset,
    )['train']
    train_dataset = train_dataset.shuffle().map(preprocess_function, num_proc=100)
    eval_dataset = load_dataset(
        'json',
        data_files=dev_dataset,
    )['train']
    eval_dataset = eval_dataset.shuffle().map(preprocess_function, num_proc=100)

    metric = evaluate.load("roc_auc")
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds)
        result = metric.compute(prediction_scores=preds, references=p.label_ids)
        return result

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=model_save_path,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=32,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            num_train_epochs=10,
            evaluation_strategy='steps',
            eval_steps=0.1,
            save_strategy='steps',
            save_steps=0.1,
            logging_strategy='steps',
            logging_steps=5,
            label_names=['labels'],
            report_to='tensorboard',
            save_total_limit=3,
            load_best_model_at_end=True,
            remove_unused_columns=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    trainer.train()
    trainer.save_model(model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["RoBERTa", "SBERT", "STEL", "LURA"], required=True)
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, required=True)
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--dev_dataset", type=str)
    parser.add_argument("--train", action='store_true', default=False)
    args = parser.parse_args()

    if args.model == "RoBERTa":
        fine_tune_RoBERTa(args.pretrained_path, args.model_save_path, args.train_dataset, args.dev_dataset)
    elif args.model == "SBERT":
        fine_tune_SBERT(args.pretrained_path, args.model_save_path, args.train_dataset, args.dev_dataset)
    elif args.model == "STEL":
        fine_tune_STEL(args.pretrained_path, args.model_save_path, args.train_dataset, args.dev_dataset)
    elif args.model == "LURA":
        fine_tune_LURA(args.pretrained_path, args.model_save_path, args.train_dataset, args.dev_dataset)
    else:
        raise ValueError("Invalid model name")
