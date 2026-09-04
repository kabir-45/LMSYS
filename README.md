# LMSYS Chatbot Arena — LLM Preference Prediction

An inference pipeline for the **LMSYS Chatbot Arena** competition, where the objective is to predict which of two anonymous language models would be preferred by a human evaluator.

The project focuses on efficient GPU inference, test-time augmentation, and reliable probability alignment for Kaggle submission.

## Overview

Given a conversation and two candidate model responses, the system predicts one of three outcomes:

- `winner_model_a`
- `winner_model_b`
- `winner_tie`

The final prediction is a probability distribution over these three classes.

### Key Features

- **Gemma 2 sequence-classification model**
- **8-bit quantized inference**
- **2× GPU data-parallel inference**
- One complete model copy per GPU
- Concurrent inference using `ThreadPoolExecutor`
- **3072-token maximum context length**
- Length-based sorting to reduce padding overhead
- Dynamic batch padding
- **Test-Time Augmentation (TTA)** using swapped model responses
- ID-based alignment of TTA predictions before averaging
- Kaggle-ready submission generation

---

## Architecture

The inference pipeline uses data parallelism rather than splitting a single model across GPUs.

```text
                         Input Conversations
                                │
                                ▼
                       Tokenization / Truncation
                                │
                                ▼
                       Sort by Sequence Length
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
              GPU 0 / model_0         GPU 1 / model_1
              Full Gemma 2 model      Full Gemma 2 model
                    │                       │
                    └───────────┬───────────┘
                                ▼
                         Base Predictions
                                │
                                ▼
                    Response-Swapped TTA
                                │
                                ▼
                      ID-Based Alignment
                                │
                                ▼
                       Probability Averaging
                                │
                                ▼
                         submission.csv
