# fgpt-instruct

This repo aims to Instruction Finetune the fgpt model through Supervised Fine Tuning. At the moment I am still experimenting with Huggingface's TRL library with pre-trained base models. Later this will be connected to the fgpt-base model.

# Current implementation 

# Results

Correct response:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Rewrite the sentence using a simile.

### Input:
The car is very fast.

Correct response:
>> The car is as fast as lightning.
```

Wrong response but still kind of close-ish. George Bernard Shaw is a playwright.
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Name the author of 'Pride and Prejudice'.

Correct response:
>> Jane Austen.

Model response:
>> Response:
The author of 'Pride and Prejudice' is George Bernard Shaw.
```