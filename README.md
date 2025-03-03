# JD2ESCO: Skill extraction from real world job ads in Chinese

This project implements a skill embedding model designed for multi-label scenarios in computer science applications. The model leverages advanced natural language processing techniques by combining a pre-trained BERT encoder, BiLSTM, and an attention mechanism. It is built with PyTorch and employs a contrastive loss function to learn robust and discriminative skill representations.

## Table of Contents

- [Overview](#overview)
- [Data Processing and Preparation](#data-processing-and-preparation)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Evaluation Metrics](#evaluation-metrics)
- [License](#license)

## Overview

The objective of this project is to generate high-quality embedding vectors for skills by processing textual data and learning representations through a combination of deep learning techniques. The workflow includes detailed data preprocessing, custom dataset construction, a dual-encoder model architecture, and robust evaluation metrics.

## Data Processing and Preparation

- **Data Loading and Preprocessing**  
  The dataset, `esco_sentence_two.csv`, is loaded using the Pandas library. The first 800 records are selected, each containing two skill IDs (`skillID_1` and `skillID_2`) along with their corresponding sentences.

- **Skill ID Extraction**  
  Unique skill IDs are extracted from both `skillID_1` and `skillID_2` fields. By flattening and deduplicating the values, a comprehensive list of unique skill IDs is obtained.

- **Mapping Skills to Sentences**  
  A dictionary is constructed to map each skill ID to its associated sentences. Sentences from both fields are merged to ensure each skill ID has an exhaustive collection of related sentences.

- **Mapping between Skill IDs and Indices**  
  Two dictionaries, `skill2idx` and `idx2skill`, are created to map each skill ID to a unique index and vice versa. This mapping facilitates efficient lookup during model training and evaluation.

- **Custom Dataset Construction**  
  A custom `SkillDataset` class (inheriting from PyTorch’s `Dataset`) is implemented to:
  - Convert raw data into a model-compatible format.
  - Generate positive samples using both skill IDs and their corresponding sentences. Skill IDs are converted to indices using `skill2idx`, and sentences are encoded using a BERT tokenizer.
  - Generate multiple negative samples per positive instance by randomly sampling sentences from skills that do not match the current positive sample. This approach improves the model’s ability to distinguish between different skills.

- **DataLoader Setup**  
  The dataset is split into 80% training and 20% validation sets. PyTorch’s DataLoader is used to batch the data, with training batches shuffled for enhanced generalization and validation batches kept in sequence.

## Model Architecture

The proposed model is a dual-encoder (BiEncoder) that consists of the following components:

1. **Base Encoder (BERT)**  
   A pre-trained Chinese BERT model (`bert-base-chinese`) is used as the foundational text encoder. It provides strong language understanding capabilities.

2. **Linear Transformation and Normalization**  
   - A Dropout layer is applied to the BERT pooler output to mitigate overfitting.
   - A fully connected layer reduces the high-dimensional BERT output to a specified embedding dimension (e.g., 128).
   - L2 normalization is applied to the embedding vectors to ensure consistency in similarity calculations.

3. **BiLSTM Layer**  
   A bidirectional LSTM is added on top of the BERT hidden states to capture contextual information, further refining the sentence representations.

4. **Attention Mechanism**  
   An attention layer is applied to the BiLSTM outputs to focus on the most informative parts of the sentence, enhancing the context-aware representation.

5. **Final Fully Connected Layer and Normalization**  
   A subsequent fully connected layer reduces the contextual vector to the target embedding dimension, followed by L2 normalization.

6. **Contrastive Loss Function**  
   The contrastive loss function is designed to:
   - Pull positive sample pairs (with similar skill embeddings) closer together.
   - Push negative sample pairs (from different skills) apart by enforcing a margin (e.g., 0.5) using a clamping operation.
   - Average the loss across all positive and negative samples to update model parameters effectively.

## Training Strategy

- **Random Seed Setting**  
  Fixed random seeds are established across Python, NumPy, and PyTorch to ensure reproducibility of experimental results.

- **Optimizer and Learning Rate**  
  The model is optimized using the AdamW optimizer with a learning rate of 2e-5, a common choice for fine-tuning BERT-based architectures.

- **Training Loop**  
  During each training epoch:
  - Both positive and negative sample embeddings are computed.
  - The contrastive loss is calculated and backpropagated to update the model parameters.
  - At the end of each epoch, evaluation is performed on the validation set using metrics such as Mean Reciprocal Rank (MRR) and Recall Precision at 5 (RP@5). The best-performing model parameters are saved.

## Evaluation Metrics

- **Mean Reciprocal Rank (MRR)**  
  MRR measures the average reciprocal rank of the correct skill in the recommendation list, reflecting the model's precision in ranking.

- **Recall Precision at 5 (RP@5)**  
  RP@5 assesses whether the correct skill appears within the top 5 recommendations, indicating the model's recall capability.

- **Evaluation Process**  
  - Representative sentence embeddings are generated for each skill ID.
  - For each validation sample, cosine similarity between its embedding and all skill embeddings is computed.
  - The skills are ranked according to similarity, and MRR and RP@5 are calculated based on the position of the correct skill.

## License

This project is licensed under the [MIT License](LICENSE).
