from gpt_download import download_and_load_gpt2 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 
   
import numpy as np

import tiktoken

from model import GPTModel

settings, params = download_and_load_gpt2(
    model_size='124M', models_dir='gpt2'
)

print("Setting:", settings)
print("Parameter dictionary keys:", params.keys())

print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)

GPT_CONFIG_124M = {
    "vocab_size": 50257, 
    "context_length": 256, 
    "emb_dim": 768, 
    "n_heads": 12, 
    "n_layers": 12, 
    "drop_rate": 0.1, 
    "qkv_bias": False 
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}, 
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, 
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20}, 
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
}

model_name = "gpt2-small (124M)" 
NEW_CONFIG = GPT_CONFIG_124M.copy() 
NEW_CONFIG.update(model_configs[model_name])
print(NEW_CONFIG)

NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()

def assign(left, right): 
    if left.shape != right.shape: 
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                         "Right: {right.shape}")
    
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params): 
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params['blocks'])): 
        q_w, k_w, v_w = np.split(
            (params['blocks'][b]['attn']['c_attn'])['w'], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        
        q_b, k_b, v_b = np.split(
            (params['blocks'][b]['attn']['c_attn'])['b'], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
        
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params['blocks'][b]['attn']['c_proj']['w'].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
        gpt.trf_blocks[b].att.out_proj.bias,
        params["blocks"][b]["attn"]["c_proj"]["b"])
        
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
        gpt.trf_blocks[b].ff.layers[0].weight,
        params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
        gpt.trf_blocks[b].ff.layers[0].bias,
        params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
        gpt.trf_blocks[b].ff.layers[2].weight,
        params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
        gpt.trf_blocks[b].ff.layers[2].bias,
        params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_blocks[b].norm1.scale = assign(
        gpt.trf_blocks[b].norm1.scale,
        params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
        gpt.trf_blocks[b].norm1.shift,
        params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
        gpt.trf_blocks[b].norm2.scale,
        params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
        gpt.trf_blocks[b].norm2.shift,
        params["blocks"][b]["ln_2"]["b"])
        
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    
load_weights_into_gpt(gpt, params)
device = torch.device('cuda' if torch.cuda.is_available() 
                      else 'mps' if torch.backends.mps.is_available() 
                      else 'cpu')
print(device)
gpt.to(device)

torch.manual_seed(123)
def generate(model, idx, max_new_tokens, context_size, top_k, temperature): 
    for _ in range(max_new_tokens): 
        idx_cond = idx[:, -context_size:]
        with torch.no_grad(): 
            logits = model(idx_cond) 
        logits = logits[:, -1, :]
        if top_k is not None: 
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1] 
            logits = torch.where(
                logits < min_val, 
                torch.tensor(float('-inf')).to(logits.device), 
                logits
            )
        if temperature > 0.0: 
            logits = logits /temperature 
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor 

def token_ids_to_text(token_ids, tokenizer): 
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

tokenizer = tiktoken.get_encoding("gpt2") 

token_ids = generate(
    model=gpt, 
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device), 
    max_new_tokens=25, 
    context_size=NEW_CONFIG["context_length"], 
    top_k=50, 
    temperature=1.5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

### chapter 6 
import urllib.request 
import zipfile 
import os 
from pathlib import Path 

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(
    url, zip_path, extracted_path, data_file_path): 
    if data_file_path.exists(): 
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return 
    
    with urllib.request.urlopen(url) as response: 
        with open(zip_path, "wb") as out_file: 
            out_file.write(response.read())
    
    with zipfile.ZipFile(zip_path, "r") as zip_ref: 
        zip_ref.extractall(extracted_path) 
        
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path) 
    print(f"File downloaded and saved as {data_file_path}")
    
download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

import pandas as pd 
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["label", "text"]
)
df

print(df['label'].value_counts())

def create_balanced_dataset(df): 
    num_spam = df[df['label'] == "spam"].shape[0]
    ham_subset = df[df['label'] == 'ham'].sample(
        num_spam, random_state=123)
    # balanced_df = pd.concat([ham_subset, df[df['label'] == 'spam']])
    return pd.concat([ham_subset, df[df['label'] == 'spam']])

balanced_df = create_balanced_dataset(df)
print(balanced_df['label'].value_counts())

balanced_df['label'] = balanced_df['label'].map({'ham': 0, 'spam': 1})

def random_split(df ,train_frac, validation_frac): 
    
    df = df.sample(
        frac=1, random_state=123
    ).reset_index(drop=True)
    train_end = int(len(df) * train_frac) 
    validation_end = train_end + int(len(df) * validation_frac) 
    
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    
    return train_df, validation_df, test_df 

train_df, validation_df, test_df = random_split(
    balanced_df, 0.7, 0.1
)

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

import torch 
from torch.utils.data import Dataset 

class SpamDataset(Dataset): 
    def __init__(self, csv_file, tokenizer, max_length=None, 
                 pad_token_id=50256): 
        self.data = pd.read_csv(csv_file)
        
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data['text']
        ]
        
        if max_length is None: 
            self.max_length = self._longest_encoded_length()
        else: 
            self.max_length = max_length 
            
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
            
        self.encoded_texts = [
            encoded_text + [pad_token_id] * 
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    
    def __len__(self): 
        return len(self.data) 
    
    def __getitem__(self, index): 
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]['label'] 
        return (
            torch.tensor(encoded, dtype=torch.long), 
            torch.tensor(label, dtype=torch.long)
        )
        
    def _longest_encoded_length(self): 
        max_length = 0 
        for encoded_text in self.encoded_texts: 
            encoded_length = len(encoded_text) 
            if encoded_length > max_length: 
                max_length = encoded_length 
        return max_length 
    
train_dataset = SpamDataset(
    csv_file='train.csv', 
    max_length=None, 
    tokenizer=tokenizer 
)
print(len(train_dataset.encoded_texts))

print(train_dataset.max_length)

val_dataset = SpamDataset(
    csv_file='validation.csv', 
    max_length=train_dataset.max_length, 
    tokenizer=tokenizer 
)
test_dataset = SpamDataset(
    csv_file='test.csv', 
    max_length=train_dataset.max_length, 
    tokenizer=tokenizer 
)

print(val_dataset.max_length)
print(test_dataset.max_length)

from torch.utils.data import DataLoader

num_workers = 0 
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers, 
    drop_last=True, 
)
val_loader = DataLoader(
    dataset=val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers, 
    drop_last=False, 
)
test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers, 
    drop_last=False,
)

for input_batch, target_batch in train_loader: 
    print("Input batch dimensions:", input_batch.shape)
    print("Target batch dimensions:", target_batch.shape)
    break

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

CHOOSE_MODEL = "gpt2-small (124M)"  
INPUT_PROMT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257, 
    "context_length": 1024, 
    "drop_rate": 0.0, 
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)" : {"emb_dim": 768, "n_layers": 12, "n_heads": 12}, 
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, 
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20}, 
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
print(model_size)
settings, params = download_and_load_gpt2(
model_size=model_size, models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

def generate_text_simple(model, idx, max_new_tokens, context_size): 
    for _ in range(max_new_tokens): 
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad(): 
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model, 
    idx=text_to_token_ids(text_1, tokenizer), 
    max_new_tokens=25, 
    context_size=BASE_CONFIG["context_length"],
)
print(token_ids_to_text(token_ids, tokenizer))

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model, 
    idx=text_to_token_ids(text_2, tokenizer), 
    max_new_tokens=23, 
    context_size=BASE_CONFIG["context_length"],
)
print(token_ids_to_text(token_ids, tokenizer))

for param in model.parameters(): 
    param.requires_grad = False 

torch.manual_seed(123)
num_classes = 2 
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"], 
    out_features=num_classes
)

for param in model.trf_blocks[-1].parameters(): 
    param.requires_grad = True 
for param in model.final_norm.parameters(): 
    param.requires_grad = True
    
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape)

with torch.no_grad(): 
    outputs = model(inputs)
print("Outputs:\n", outputs)
print("Outputs dimension:", outputs.shape)

print("Last outpout token:", outputs[:, -1, :])

probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas, dim=-1)
print("Class label:", label.item())

logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("Class label:", label.item())

def calc_accuracy_loader(data_loader, model, device, num_batches=None): 
    
    model.eval() 
    correct_predictions, num_examples = 0, 0 
    
    if num_batches is None: 
        num_batches = len(data_loader)
    else: 
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader): 
        if i < num_batches: 
            input_batch = input_batch.to(device) 
            target_batch = target_batch.to(device) 
            
            with torch.no_grad(): 
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
    
        else:
            break
    return correct_predictions / num_examples 

device = torch.device('cuda' if torch.cuda.is_available() 
                      else 'mps' if torch.backends.mps.is_available() 
                      else 'cpu')
print(device)
model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)
val_accuracy = calc_accuracy_loader(
    val_loader, model, device, num_batches=10
)
test_accuracy = calc_accuracy_loader(
    test_loader, model, device, num_batches=10
)

print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

def calc_loss_batch(input_batch, target_batch, model, device): 
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    # loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return torch.nn.functional.cross_entropy(logits, target_batch)

def calc_loss_loader(data_loader, model, device, num_batches=None): 
    total_loss = 0.0 
    if len(data_loader) == 0: 
        return float("nan") 
    elif num_batches is None: 
        num_batches = len(data_loader)
    else: 
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader): 
        if i < num_batches: 
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break 
    return total_loss / num_batches 

with torch.no_grad(): 
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")

def train_classifier_simple(
    model, train_loader, val_loader, optimizer, device, 
    num_epochs, eval_freq, eval_iter):
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1 
    
    for epoch in range(num_epochs): 
        model.train() 
        
        for input_batch, target_batch in train_loader: 
            optimizer.zero_grad() 
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() 
            optimizer.step() 
            
            examples_seen += input_batch.shape[0]
            global_step += 1 
            
            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss) 
                val_losses.append(val_loss) 
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")
        
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter)
        
        print(f"Training accuracy: {train_accuracy * 100:.2f} | ", end="")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        
    return train_losses, val_losses, train_accs, val_accs, examples_seen 

def evaluate_model(model, train_loader, val_loader, device, eval_iter): 
    model.eval() 
    with torch.no_grad(): 
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train() 
    return train_loss, val_loss

import time 

device = torch.device('cuda' if torch.cuda.is_available() 
                      else 'mps' if torch.backends.mps.is_available() 
                      else 'cpu')
print(device)
model.to(device)

start_time = time.time() 
torch.manual_seed(123) 
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1) 
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,  
        num_epochs=num_epochs, eval_freq=50, 
        eval_iter=5
    )
    
end_time = time.time() 
execution_time_minutes = (end_time - start_time) / 60 
print(f"Training completed in {execution_time_minutes:.2f} minutes")

import matplotlib.pyplot as plt 

def plot_values(
    epochs_seen, examples_seen, train_values, val_values, 
    label='loss'):
    
    fig, ax1 = plt.subplots(figsize=(5, 3))
    
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle='-.', 
        label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend() 
    
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
    
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label='accuracy')

train_accuracy = calc_accuracy_loader(train_loader, model, device) 
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy * 100:.2f}")
print(f"Validation accuracy: {val_accuracy * 100:.2f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}")

def classify_review(
    text, model, tokenizer, device, max_length=None, 
    pad_token_id=50256): 
    model.eval() 
    
    input_ids = tokenizer.encode(text) 
    supported_context_length = model.pos_emb.weight.shape[1]
    
    input_ids = input_ids[:min(
        max_length, supported_context_length
    )]
    
    input_ids += [pad_token_id] * ( max_length - len(input_ids))
    
    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)
    
    with torch.no_grad(): 
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item() 
    
    return "spam" if predicted_label == 1 else "not spam"

text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify_review(text_1, model, tokenizer, device, max_length=train_dataset.max_length))

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(classify_review(text_2, model, tokenizer, device, max_length=train_dataset.max_length))

torch.save(model.state_dict(), "review_classifier.pth")

model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True)
model.load_state_dict(model_state_dict)