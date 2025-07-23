Since I'm just starting to learn about LLMs and model training, I'll use this file to document my understanding of some new concepts and terminology.

## Tokens and Tokenization
Breaking text into chunks that can be fed into a model. For example: "Dogs are the best!" might become
```
["Dogs", " are", " the", "best", "!"]
```

## Transformer
A type of neural network architecture that helps a model understand the relationship between words.

## Model Weights
Learned patterns from data that inform the model about things like:
- How much attention to pay to each word
- What the next word should be
- What makes a good sentence

## Epochs
An epoch is one full pass through the training dataset.

A model can improve with successive epochs, up to a point where it may start to overfit.

## Fine-tuning
A process we can apply to a foundational model to help it get better at a specific task, such as speaking like you in your whatsapp group.

## LoRA (Low Rank Adaptation)
A process to make fine-tuning more efficient, since training is computationally intensive. It "freezes" most of the model and adds something called "adapters" that tweak behaviour without retraining the whole thing.

## Quantization
A process of compressing the model weights to smaller numbers, for example going from 32-bit to 4-bit. This allows the model to use less memory and run faster on smaller machines. Sometimes it impacts performance but not always noticeably.
