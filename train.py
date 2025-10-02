import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore
from tqdm import tqdm
import shutil
import warnings
from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import BilungualDataset, causal_mask
from model import build_transformer

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encode output and reuse it for every step
    encode_output = model.encode(source, source_mask)
    
    # Initialize the decode input with the sos token
    decode_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decode_input.size(1) == max_len:
            break
            
        # Build the decode mask for the current sequence length
        decode_mask = causal_mask(decode_input.size(1)).type_as(source_mask).to(device) #(1, 1, seq_len, seq_len)
        
        # Get decode output
        out = model.decode(
            encode_output,                                  # memory from encode
            source_mask,  # embedded decode input
            decode_input,                              # source mask for cross-attention
            decode_mask                                    # causal mask for self-attention
        )
        
        # Get the next token
        prob = model.project(out[:, -1])  # (1, vocab_size)
        _, next_word = torch.max(prob, dim=1)
        
        decode_input = torch.cat(
            [decode_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1

        )
        
        if next_word == eos_idx:
            break
            
    return decode_input.squeeze(0)

def run_validation(model_for_ops, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model_for_ops.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        console_width = shutil.get_terminal_size().columns
    except Exception:
        console_width = 80

    # try:
    #     # get the console window width
    #     with os.popen('stty size', 'r') as console:
    #         _, console_width = console.read().split()
    #         console_width = int(console_width)
    # except:
    #     # If we can't get the console width, use 80 as default
    #     console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encode_input = batch["encoder_input"].to(device) # (b, seq_len)
            encode_mask = batch["encoder_attention_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encode_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model_for_ops, encode_input, encode_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang=lang))
    if not Path.exists(tokenizer_path):
        print(f"Building tokenizer for {lang}...")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
        
def get_ds(config):
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['source_lang']}-{config['target_lang']}", split='train')
    
    # Build or load tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['source_lang'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['target_lang'])

    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilungualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['source_lang'], config['target_lang'], config['max_seq_length'])
    val_ds = BilungualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['source_lang'], config['target_lang'], config['max_seq_length'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['source_lang']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['target_lang']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max sequence length in {config['source_lang']}: {max_len_src}")
    print(f"Max sequence length in {config['target_lang']}: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

# check this if error happens
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len, vocab_tgt_len, config['max_seq_length'],
        d_model=config['d_model'],
    )
    return model

def train_model(config):
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        model = model.to(device)

    # ...after model creation...
    if isinstance(model, nn.DataParallel):
        model_for_ops = model.module
    else:
        model_for_ops = model

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1:02d}")
        for batch in batch_iterator:
            enc_inputs = batch["encoder_input"].to(device) #(B, seq_len)
            dec_inputs = batch["decoder_input"].to(device) #(B, seq_len)
            encode_mask = batch["encoder_attention_mask"].to(device) #(B, 1, 1, seq_len)
            decode_mask = batch["decoder_attention_mask"].to(device) #(B, 1, seq_len, seq_len)

            # Run the tensors through the model
            encode_output = model_for_ops.encode(enc_inputs, encode_mask) #(B, seq_len, d_model)
            decode_output = model_for_ops.decode(
                encode_output,                                 
                encode_mask,
                dec_inputs,                                
                decode_mask                           
            ) #(B, seq_len, d_model)
            proj_output = model_for_ops.project(decode_output)

            labels = batch["label"].to(device) #(B, seq_len)
            # (B*seq_len, vocab_tgt_size), (B*seq_len)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), labels.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            run_validation(model_for_ops, val_dataloader, tokenizer_src, tokenizer_tgt, config['max_seq_length'], device, lambda msg: batch_iterator.write(msg), global_step, writer, num_examples=2)

        # Save model checkpoint
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    config = get_config()
    train_model(config)
