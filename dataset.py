import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class BilungualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_seq_length = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        src_text = item['translation'][self.src_lang]
        tgt_text = item['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Pad sequences
        enc_num_padding_tokens = self.max_seq_length - len(enc_input_tokens) - 2  # for SOS and EOS
        dec_num_padding_tokens = self.max_seq_length - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sequence length exceeded max_seq_length of {self.max_seq_length}")
        
        # Add SOS and EOS tokens
        enc_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(enc_input_tokens, dtype=torch.int64), 
                self.eos_token,
                torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )
        #remove .item() from pad token
        dec_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert enc_input.size(0) == self.max_seq_length
        assert dec_input.size(0) == self.max_seq_length
        assert label.size(0) == self.max_seq_length

        return {
            'encoder_input': enc_input,
            'decoder_input': dec_input,
            'encoder_attention_mask': (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_attention_mask': (dec_input != self.pad_token).unsqueeze(0).int() & causal_mask(dec_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            'src_text': src_text,
            'tgt_text': tgt_text,
            "label": label
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0