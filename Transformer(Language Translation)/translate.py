import torch
from config import get_config, get_weights_file_path
from model import build_transformer
from dataset import get_all_sentences, get_or_build_tokenizer
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.nn.functional import log_softmax
from dataset import causal_mask
from dataset import get_all_sentences, get_or_build_tokenizer

def greedy_decode(model, src, src_mask, tokenizer_tgt, max_len, device):
    sos_id = tokenizer_tgt.token_to_id("[SOS]")
    eos_id = tokenizer_tgt.token_to_id("[EOS]")

    # Start with SOS token
    decoder_input = torch.tensor([[sos_id]], device=device)
    
    for _ in range(max_len):
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        out = model.decode(src, src_mask, decoder_input, decoder_mask)
        out = model.project(out)
        next_token = out[:, -1].argmax(-1).unsqueeze(0)
        decoder_input = torch.cat([decoder_input, next_token], dim=1)

        if next_token.item() == eos_id:
            break

    return decoder_input.squeeze(0)

def translate_sentence(model, sentence, tokenizer_src, tokenizer_tgt, config, device):
    model.eval()
    tokens = tokenizer_src.encode(sentence).ids
    tokens = tokens[:config["seq_len"] - 2]  # reserve space for [SOS] and [EOS]

    input_ids = [tokenizer_src.token_to_id("[SOS]")] + tokens + [tokenizer_src.token_to_id("[EOS]")]
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = (input_tensor != tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).int()

    encoded = model.encode(input_tensor, src_mask)
    decoded_ids = greedy_decode(model, encoded, src_mask, tokenizer_tgt, config["seq_len"], device)
    decoded_ids = decoded_ids.tolist()

    # Remove special tokens
    output_tokens = [
        idx for idx in decoded_ids
        if idx not in {
            tokenizer_tgt.token_to_id("[PAD]"),
            tokenizer_tgt.token_to_id("[SOS]"),
            tokenizer_tgt.token_to_id("[EOS]")
        }
    ]
    return tokenizer_tgt.decode(output_tokens)

if __name__ == "__main__":
    config = get_config()
    config["preload"] = "manual_save"

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    print(f"Using device: {device}")

    ds_raw = load_dataset(config["datasource"], split="train")
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"]
    ).to(device)

    model_path = get_weights_file_path(config, config["preload"])
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    print(f"Loaded model from: {model_path}")

    # 🔤 Translate a sentence
    while True:
        sentence = input("\nEnter sentence (or 'quit'): ").strip()
        if sentence.lower() in ["quit", "exit"]:
            break
        translated = translate_sentence(model, sentence, tokenizer_src, tokenizer_tgt, config, device)
        print(f"Translated: {translated}")
