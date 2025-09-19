import torch
import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pathlib import Path
import pandas as pd

def _create_dummy_dataset(name, max_samples):
    # Simulate loading for datasets not on Hugging Face hub
    data = {
        'prompt': [f'This is a dummy prompt {i} for {name}.' for i in range(max_samples)],
        'bad_response': [f'This is a bad response {i}.' for i in range(max_samples)],
        'good_response': [f'This is a much better and factual response {i}.' for i in range(max_samples)]
    }
    return Dataset.from_dict(data)

def _precache_features(dataset, config):
    print("Precaching features...")
    # 1. Embeddings with e5-large-v2 (or a smaller model for smoke test)
    embedding_model = SentenceTransformer(config['preprocessing']['embedding_model'])
    good_responses = [item['good_response'] for item in dataset]
    bad_responses = [item['bad_response'] for item in dataset]
    good_embeddings = embedding_model.encode(good_responses, convert_to_tensor=True, show_progress_bar=True)
    bad_embeddings = embedding_model.encode(bad_responses, convert_to_tensor=True, show_progress_bar=True)

    # 2. FactualityBERT logits (simulated)
    fact_good_logits = torch.rand(len(dataset), good_embeddings.shape[1])
    fact_bad_logits = torch.rand(len(dataset), bad_embeddings.shape[1])

    # 3. TF-IDF weights
    tfidf_path = Path(config['preprocessing']['tfidf_path'])
    if tfidf_path.exists():
        with open(tfidf_path, 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        vectorizer = TfidfVectorizer(min_df=5, lowercase=True)
        all_text = [item.lower() for item in good_responses + bad_responses]
        vectorizer.fit(all_text)
        tfidf_path.parent.mkdir(exist_ok=True, parents=True)
        with open(tfidf_path, 'wb') as f:
            pickle.dump(vectorizer, f)

    good_tfidf_sparse = vectorizer.transform(good_responses)
    bad_tfidf_sparse = vectorizer.transform(bad_responses)
    good_tfidf = torch.tensor(good_tfidf_sparse.toarray(), dtype=torch.float32)
    bad_tfidf = torch.tensor(bad_tfidf_sparse.toarray(), dtype=torch.float32)

    # This part is a placeholder. Real TF-IDF needs to be mapped to token-level.
    # For simplicity, we'll average it for now.
    good_tfidf_token_level = good_tfidf.mean(dim=1, keepdim=True).expand(-1, good_embeddings.shape[1])
    bad_tfidf_token_level = bad_tfidf.mean(dim=1, keepdim=True).expand(-1, bad_embeddings.shape[1])

    dataset = dataset.add_column("good_emb", good_embeddings.cpu().numpy().tolist())
    dataset = dataset.add_column("bad_emb", bad_embeddings.cpu().numpy().tolist())
    dataset = dataset.add_column("fact_good", fact_good_logits.cpu().numpy().tolist())
    dataset = dataset.add_column("fact_bad", fact_bad_logits.cpu().numpy().tolist())
    dataset = dataset.add_column("tfidf_good", good_tfidf_token_level.cpu().numpy().tolist())
    dataset = dataset.add_column("tfidf_bad", bad_tfidf_token_level.cpu().numpy().tolist())

    return dataset

def load_and_prepare_data(config, tokenizer):
    all_data = []
    for ds_config in config['datasets']:
        if 'custom' in ds_config['name']:
            dataset = _create_dummy_dataset(ds_config['name'], ds_config['max_samples'])
        elif ds_config['name'] == 'zhihanyang/sharegpt-cleaned':
            temp_ds = load_dataset(ds_config['name'], split=ds_config['split']).select(range(ds_config['max_samples']))
            # Adapt format
            data = {
                'prompt': temp_ds['prompt'],
                'good_response': temp_ds['response'],
                'bad_response': ['Placeholder bad response'] * len(temp_ds)
            }
            dataset = Dataset.from_dict(data)
        else:
             # This is a simplification; other datasets require specific formatting
            dataset = _create_dummy_dataset(ds_config['name'], ds_config['max_samples'])
        all_data.append(dataset.to_pandas())

    combined_df = pd.concat(all_data, ignore_index=True)
    train_dataset = Dataset.from_pandas(combined_df)
    
    print("Precaching features...")
    try:
        train_dataset = _precache_features(train_dataset, config)
    except Exception as e:
        print(f"Warning: Could not precache features: {e}")
        print("Continuing without precached features...")

    def tokenize(examples):
        prompts = examples['prompt']
        good_res = examples['good_response']
        bad_res = examples['bad_response']
        
        # For training, combine prompt and good response
        full_text = [p + " " + gr for p, gr in zip(prompts, good_res)]
        tokenized_inputs = tokenizer(full_text, truncation=True, padding='max_length', max_length=config['preprocessing']['max_length'])
        
        tokenized_good = tokenizer(good_res, truncation=True, padding='max_length', max_length=config['preprocessing']['max_length'])
        tokenized_bad = tokenizer(bad_res, truncation=True, padding='max_length', max_length=config['preprocessing']['max_length'])

        # Add necessary fields for loss calculation
        tokenized_inputs['good_ids'] = tokenized_good['input_ids']
        tokenized_inputs['bad_ids'] = tokenized_bad['input_ids']
        tokenized_inputs['good_ids_len'] = [len(x) for x in tokenized_good['input_ids']]
        tokenized_inputs['bad_ids_len'] = [len(x) for x in tokenized_bad['input_ids']]

        return tokenized_inputs

    train_dataset = train_dataset.map(tokenize, batched=True)
    
    available_columns = train_dataset.column_names
    required_columns = ['input_ids', 'attention_mask', 'good_ids', 'bad_ids', 'good_ids_len', 'bad_ids_len']
    optional_columns = ['good_emb', 'bad_emb', 'fact_good', 'fact_bad', 'tfidf_good', 'tfidf_bad']
    
    format_columns = required_columns + [col for col in optional_columns if col in available_columns]
    
    train_dataset.set_format(type='torch', columns=format_columns)
    
    # Prepare eval dataset
    eval_ds_config = config['evaluation']['eval_datasets'][0]
    try:
        if eval_ds_config['name'] in ['truthfulqa', 'truthful_qa']:
            eval_dataset = load_dataset('truthful_qa', 'generation', split=eval_ds_config['split']).select(range(eval_ds_config['max_samples']))
        else:
            eval_dataset = load_dataset(eval_ds_config['name'], eval_ds_config.get('subset'), split=eval_ds_config['split']).select(range(eval_ds_config['max_samples']))
    except Exception as e:
        print(f"Warning: Could not load eval dataset {eval_ds_config['name']}: {e}")
        print("Creating dummy eval dataset...")
        eval_data = {
            'question': [f'What is the answer to question {i}?' for i in range(eval_ds_config['max_samples'])],
            'best_answer': [f'This is the best answer {i}.' for i in range(eval_ds_config['max_samples'])]
        }
        eval_dataset = Dataset.from_dict(eval_data)
    
    def tokenize_eval(examples):
        question_key = 'question' if 'question' in examples else 'prompt'
        answer_key = 'best_answer' if 'best_answer' in examples else 'answer'
        
        questions = examples.get(question_key, examples.get('prompt', [''] * len(examples.get('question', ['']))))
        answers = examples.get(answer_key, examples.get('response', [''] * len(questions)))
        
        inputs = tokenizer(questions, truncation=True, padding='max_length', max_length=config['preprocessing']['max_length'])
        inputs['reference_text'] = answers
        return inputs

    eval_dataset = eval_dataset.map(tokenize_eval, batched=True, remove_columns=eval_dataset.column_names)
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'reference_text'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['training_args']['per_device_train_batch_size'])
    eval_dataloader = DataLoader(eval_dataset, batch_size=config['evaluation']['per_device_eval_batch_size'])

    return train_dataloader, eval_dataloader
