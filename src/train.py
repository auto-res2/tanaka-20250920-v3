import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
import pot as ot

def soft_far_loss(logits, hidden_states, batch, soft_far_params, device):
    good_emb = batch['good_emb'].to(device)
    bad_emb = batch['bad_emb'].to(device)
    fact_good = batch['fact_good'].to(device)
    fact_bad = batch['fact_bad'].to(device)
    tfidf_good = batch['tfidf_good'].to(device)
    tfidf_bad = batch['tfidf_bad'].to(device)
    good_ids = batch['good_ids'].to(device)
    bad_ids = batch['bad_ids'].to(device)

    # A. Soft token alignment
    S = torch.einsum('bid,bjd->bij', good_emb, bad_emb) / soft_far_params['sinkhorn_epsilon']
    
    # Assuming batch processing for sinkhorn. This is a simplification.
    # In a real scenario, this might need a loop or a batched sinkhorn implementation.
    A = torch.zeros_like(S)
    for i in range(S.size(0)):
        try:
            A[i] = ot.sinkhorn(
                torch.ones(good_emb.size(1), device=device) / good_emb.size(1),
                torch.ones(bad_emb.size(1), device=device) / bad_emb.size(1),
                S[i]
            )
        except Exception as e:
            # Sinkhorn may fail to converge, handle gracefully
            print(f"Sinkhorn convergence failed: {e}. Using identity-like matrix.")
            min_dim = min(good_emb.size(1), bad_emb.size(1))
            A[i, :min_dim, :min_dim] = torch.eye(min_dim, device=device)

    # C. Loss function
    probs = torch.nn.functional.log_softmax(logits, dim=-1)
    good_probs = torch.gather(probs, -1, good_ids.unsqueeze(-1)).squeeze(-1)
    bad_probs = torch.gather(probs, -1, bad_ids.unsqueeze(-1)).squeeze(-1)

    q = soft_far_params['lambda_fact'] * (fact_good - torch.bmm(A, fact_bad.unsqueeze(-1)).squeeze(-1)) + (1 - soft_far_params['lambda_fact']) * tfidf_good
    
    loss_pos = -torch.mean(torch.bmm(A, q.unsqueeze(-1)).squeeze(-1) * good_probs)

    unmatched_bad_mass = 1 - A.sum(dim=1)
    loss_neg = -torch.mean(unmatched_bad_mass * tfidf_bad * bad_probs)

    # Representation Wasserstein distance (simplified as squared Euclidean distance of mean representations)
    hid_good = hidden_states[torch.arange(hidden_states.size(0)), batch['good_ids_len']-1]
    hid_bad = hidden_states[torch.arange(hidden_states.size(0)), batch['bad_ids_len']-1]
    w2 = torch.sqrt(torch.sum((hid_good - hid_bad) ** 2, dim=-1)).mean()

    total_loss = loss_pos + loss_neg + soft_far_params['delta_w2'] * w2
    return total_loss

def get_model_and_tokenizer(model_name, fp16=True):
    token = os.getenv('HF_TOKEN')
    dtype = torch.float16 if fp16 else torch.bfloat16
    if not torch.cuda.is_available():
        dtype = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            token=token,
            trust_remote_code=True 
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise

    return model, tokenizer

def train_model(config, model, tokenizer, train_dataloader, eval_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['training_args']['learning_rate'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training_args']['warmup_steps'],
        num_training_steps=config['training_args']['max_steps']
    )

    model.train()
    pbar = tqdm(total=config['training_args']['max_steps'], desc="Training")
    global_step = 0
    train_losses = []

    while global_step < config['training_args']['max_steps']:
        for batch in train_dataloader:
            if global_step >= config['training_args']['max_steps']:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            loss = soft_far_loss(
                outputs.logits,
                outputs.hidden_states[-1],
                batch,
                config['soft_far_params'],
                device
            )
            
            loss = loss / config['training_args']['gradient_accumulation_steps']
            loss.backward()

            if (global_step + 1) % config['training_args']['gradient_accumulation_steps'] == 0:
                clip_grad_norm_(model.parameters(), config['training_args']['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_losses.append(loss.item() * config['training_args']['gradient_accumulation_steps'])
            pbar.update(1)
            pbar.set_postfix({"loss": train_losses[-1]})
            global_step += 1

    pbar.close()
    return model, train_losses
