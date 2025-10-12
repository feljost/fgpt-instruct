import json
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tiktoken
from transformers import GPT2LMHeadModel



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
    ignore_token_id:int = -100,  # for xent loss, -100 is ignored by default
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
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_batched = []
    targets_batched = []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [eos_token_id]
        # Pad sequences to batch_max_length
        padded = (
            new_item + [eos_token_id] *
            (batch_max_length - len(new_item))
        )
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
    drop_last=True, # only send full batches
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=create_padded_batch,
    shuffle=True,
    drop_last=True, # only send full batches
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=create_padded_batch,
    shuffle=True,
    drop_last=True, # only send full batches
)


print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)


# go on from here... so far I only had time for the data pre-processing part




# # Load GPT-2 Medium model
# model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to("cuda")
# print("Loaded GPT-2 Medium model.")

# # Generate results from GPT-2 Medium model on the test set
# model.eval()
# results = []

# with torch.no_grad():
#     for inputs, _ in test_loader:
#         outputs = model.generate(
#             input_ids=inputs,
#             max_length=inputs.shape[1] + 50,
#             do_sample=True,
#             top_k=50,
#             top_p=0.95,
#             eos_token_id=50256,
#             pad_token_id=50256,
#         )
#         for i in range(outputs.size(0)):
#             decoded = tokenizer.decode(outputs[i].cpu().numpy())
#             results.append(decoded)

# # Print a few generated results
# for i, res in enumerate(results[:5]):
#     print(f"Sample {i+1}:\n{res}\n{'-'*40}")

# print("...")