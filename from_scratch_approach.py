import json
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tiktoken
from transformers import GPT2LMHeadModel
from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch05 import download_and_load_gpt2
from llms_from_scratch.ch05 import load_weights_into_gpt
from llms_from_scratch.ch05 import generate
from llms_from_scratch.ch05 import train_model_simple


# load data  ( TODO: put this in a function / class )
with open("data/instruction_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Data loaded with {len(data)} entries")


train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)  # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion : train_portion + test_portion]
val_data = data[train_portion + test_portion :]


print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

# create tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
# TODO: Ensure the special tokens align with the previous fgpt-base


def format_input(entry):
    """Turns the given text input into Alpaca style input.
    This means that for any prompt, it will turn it into a bigger prompt,
    that describes a task."""
    # TODO: Perhaps adjus this for a different method (like Phi-3 style)
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        self.encoded_texts = []
        for entry in data:
            # put it into the alpaca style format
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            # encode into tokens
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        """Allows to index (e.g. data[0] or data[0:3]) to return
        tokenized representation of data"""
        return self.encoded_texts[index]

    def __len__(self):
        """Needed for DataLoader from torch later"""
        return len(self.data)


def create_padded_batch(
    batch: list[list[int]],
    eos_token_id: int = 50256,
    ignore_token_id: int = -100,  # for xent loss, -100 is ignored by default
):
    """
    This function creates the batches for training.
    It takes the batch as an input, then adds a padding / eos token at the end, and
    fills the rest with the ingore_token_id.

    args:
        batch: list of lists, where each sublist is a sequence of token ids
        eos_token_id: the token id used for padding (usually the eos token)
        ignore_token_id: the token id used to mask out the
            in the loss calculation (usually -100 for xent loss)

    returns:
        inputs_tensor: tensor of shape (batch_size, max_seq_length) with input ids
        targets_tensor: tensor of shape (batch_size, max_seq_length) with target ids


    Example:
        >>> batch = [
        >>>     [0, 1, 2, 3, 4],
        >>>     [5, 6],
        >>>     [7, 8, 9]
        >>> ]
        >>> create_padded_batch(batch)
        tensor([[    0,     1,     2,     3,     4],
                [    5,     6, 50256,  -100,  -100],
                [    7,     8,     9, 50256,  -100]])
        tensor([[    1,     2,     3,     4, 50256],
                [    6, 50256,  -100,  -100,  -100],
                [    8,     9,  -100,  -100,  -100]])

    """

    # Find the longest sequence in the batch
    # and increase the max length by +1, which will add one extra
    # padding token below
    batch_max_length = max(len(item) + 1 for item in batch)

    # Pad and prepare inputs
    inputs_batched = []
    targets_batched = []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [eos_token_id]
        # Pad sequences to batch_max_length
        padded = new_item + [eos_token_id] * (batch_max_length - len(new_item))
        # Via padded[:-1], we remove the extra padded token
        # that has been added via the +1 setting in batch_max_length
        # (implementation of rasbt, sticking to it for simplicity)
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])  # shifted one to the right

        # Replace all but the first padding tokens in targets by ignore_token_id
        mask = targets == eos_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_token_id

        inputs_batched.append(inputs)
        targets_batched.append(targets)

    inputs_tensor = torch.stack(inputs_batched).to("cuda")
    targets_tensor = torch.stack(targets_batched).to("cuda")

    return inputs_tensor, targets_tensor


num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=create_padded_batch,
    shuffle=True,
    drop_last=True,  # only send full batches
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=create_padded_batch,
    shuffle=True,
    drop_last=True,  # only send full batches
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=create_padded_batch,
    shuffle=True,
    drop_last=True,  # only send full batches
)


print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)

# --------------------------------------------------------------------------------------
# This part is a placeholder.. Initially will use the llm-from-scratch code but later
# will use my own fgpt-base code
BASE_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True,  # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)  # this is a pytorch model
load_weights_into_gpt(model, params)
model.to("cuda")
model.eval()

input_text = "Hello, world!"
input_tokens = tokenizer.encode(input_text)
input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to("cuda")


token_ids = generate(
    model=model,
    idx=input_tensor,
    max_new_tokens=256,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = tokenizer.decode(token_ids[0].cpu().numpy())
response_text = generated_text.lstrip(input_text)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device="cuda",
    num_epochs=1,
    eval_freq=5,
    eval_iter=5,
    start_context=format_input(val_data[0]),
    tokenizer=tokenizer,
)

# Evaluation on test set

for entry in test_data[:3]:
    input_text = format_input(entry)
    input_tokens = tokenizer.encode(input_text)
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to("cuda")

    # todo: replace with fgpt-base generate
    token_ids = generate(
        model=model,
        idx=input_tensor,
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )

    generated_text = tokenizer.decode(token_ids[0].cpu().numpy())
    response_text = generated_text.lstrip(input_text)

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")
