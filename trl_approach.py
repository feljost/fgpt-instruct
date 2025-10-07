from trl import SFTConfig
from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline
import torch

base_model_name = "Qwen/Qwen3-0.6B-Base"


train_dataset = load_dataset("trl-lib/Capybara", split="train").select(range(500))

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Exchange '<|endoftext|>' with '<|im_end|>' for chat interface
if tokenizer.eos_token != "<|im_end|>":
    tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})
    base_model.resize_token_embeddings(len(tokenizer))

# ensure that eos and padding tokens are the same.
base_model.config.eos_token_id = tokenizer.eos_token_id
base_model.config.pad_token_id = tokenizer.pad_token_id

# Check Initial capability:
pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer)
prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one word.<|im_end|>\n<|im_start|>assistant\n"
response_text = pipe(prompt)[0]["generated_text"]
response_text = response_text.replace(prompt, "")
print("Base Model Autocompletion")
print(f"PROMPT: '{prompt}'")
print(f"RESPONSE: '{response_text}'")


# Actual SFT Training
trainer = SFTTrainer(
    model=base_model,
    args=SFTConfig(
        output_dir="Qwen3-0.6B-Instruct",
        chat_template_path="HuggingFaceTB/SmolLM3-3B",
        eos_token="<|im_end|>",
    ),
    train_dataset=train_dataset,
)
trainer.train()
# this will save the model in output dir and not keep it in memory

del base_model  # free up memory

# Check results
instruct_model = AutoModelForCausalLM.from_pretrained(
    "Qwen3-0.6B-Instruct/checkpoint-39", dtype=torch.bfloat16
)
pipe = pipeline("text-generation", model=instruct_model,  tokenizer=tokenizer)
prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one word.<|im_end|>\n<|im_start|>assistant\n"
response_text = pipe(prompt)[0]["generated_text"]
response_text = response_text.replace(prompt, "")
print("Instruct Model")
print(f"PROMPT: '{prompt}'")
print(f"RESPONSE: '{response_text}'")