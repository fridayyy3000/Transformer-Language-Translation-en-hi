import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import os
import warnings

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.trainers import WordLevelTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset(config["datasource"], split="train")
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    total_size = len(ds_raw)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_raw, val_raw = random_split(ds_raw, [train_size, val_size])

    train_ds = BilingualDataset(train_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    max_len_src = max(len(tokenizer_src.encode(x['translation'][config['lang_src']]).ids) for x in ds_raw)
    max_len_tgt = max(len(tokenizer_tgt.encode(x['translation'][config['lang_tgt']]).ids) for x in ds_raw)

    print(f"Max source length: {max_len_src}")
    print(f"Max target length: {max_len_tgt}")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt

@torch.no_grad()
def run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, seq_len, device, print_msg, global_step, writer):
    model.eval()
    total_loss = 0
    count = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1)

    for batch in val_dataloader:
        encoder_input = batch["encoder_input"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        decoder_mask = batch["decoder_mask"].to(device)
        label = batch["label"].to(device)

        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = model.project(decoder_output)

        loss = loss_fn(proj_output.view(-1, proj_output.size(-1)), label.view(-1))
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / count
    print_msg(f"Validation loss: {avg_loss:.3f}")
    writer.add_scalar("val loss", avg_loss, global_step)
    writer.flush()

def train_model(config):
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print("Using device:", device)

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    writer = SummaryWriter(config["experiment_name"])

    initial_epoch = 0
    global_step = 0
    model_path = (
        latest_weights_file_path(config)
        if config["preload"] == "latest"
        else get_weights_file_path(config, config["preload"])
        if config["preload"]
        else None
    )
    if model_path and os.path.exists(model_path):
        print(f"Preloading model {model_path}")
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        total_batches = len(train_dataloader)
        save_every = max(total_batches // 10, 1)  # every 10%

        for i, batch in enumerate(batch_iterator):
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, proj_output.size(-1)), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if (i + 1) % save_every == 0:
                checkpoint_filename = get_weights_file_path(config, f"epoch{epoch:02d}_step{i+1}")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step
                }, checkpoint_filename)
                print(f"Intermediate checkpoint saved to: {checkpoint_filename}")

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, writer)

    final_filename = get_weights_file_path(config, "manual_save")
    torch.save({
        "epoch": config["num_epochs"] - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step
    }, final_filename)
    print(f" Final model manually saved to: {final_filename}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
