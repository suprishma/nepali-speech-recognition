# Nepali Speech Recognition

Automatic Speech Recognition (ASR) for the Nepali language using a fine-tuned 
Wav2Vec2 model. Nepali is a low-resource language with limited ASR support, 
making this a challenging but important task.

## Overview

Fine-tuned `facebook/wav2vec2-xls-r-300m` (300M parameter multilingual model) 
on a Nepali speech dataset using CTC (Connectionist Temporal Classification). 
A custom Nepali character-level vocabulary was built from scratch. The pipeline 
also includes a post-processing step that automatically adds Nepali punctuation 
(पूर्णविराम) using a verb detection system (kriyapad).

## Results

**Training (Step 3000)**
| Metric | Score |
|--------|-------|
| WER | 0.392 |
| CER | 0.088 |

**Inference on custom audio**
| Metric | Score |
|--------|-------|
| WER | 0.653 |
| CER | 0.154 |

> Higher WER on inference is expected due to domain mismatch between 
> training data and custom audio input. CER of 0.154 indicates strong 
> character-level recognition.

## Tech Stack
Python, HuggingFace Transformers, Wav2Vec2, CTC, Google Colab (GPU)

## Dataset
[OpenSLR SLR143](https://www.openslr.org/143) — Nepali male and female voice 
recordings with transcriptions. Split 90% train / 10% validation.

## Training Details
- Base model: `facebook/wav2vec2-xls-r-300m`
- Epochs: 40
- Custom Nepali character vocabulary built from dataset
- Evaluation metrics: WER and CER
