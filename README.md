# Fake News Detection: Transformers and LLM (

## Project Overview
This project involves a comparative analysis of different neural network architectures for the task of **Fake News Detection**[cite: 7]. [cite_start]The primary goal is to evaluate the performance, efficiency, and architectural nuances of recurrent models versus attention-based Transformer models.

## Requirements
* **Python** 3.8+
* **PyTorch** (for model implementation)
* **Transformers** (Hugging Face library for BERT)
* **Pandas** & **NumPy** (for data manipulation)
* **Scikit-learn** (for evaluation metrics) 

## Dataset
The project utilizes the **Fake News Dataset**, which consists of:
* `True.csv`: Real-world news articles.
* `Fake.csv`: Fabricated news articles.

## Implementation Details
As per the assignment requirements, the following components were developed:

### 1. Model Architectures
* **Recurrent Baselines**: Vanilla RNN and LSTM models[cite: 11].
* **Custom Transformer Encoder**: Built using PyTorch's built-in `TransformerEncoder` modules. This includes:
    * **Token Embedding**: Converts word indices to dense vectors.
    * **Positional Encoding**: Injects sequence order information.
    * **Padding Masking**: Handles variable sequence lengths during attention
    * **Pooling Method**: Aggregates encoder outputs for the final layer.
* **Pretrained BERT**: A fine-tuned `bert-base-uncased` model.

### 2. Design Studies
Two design studies were conducted to analyze Transformer behavior:
* **Positional Encoding**: Evaluation of the model with vs. without positional encodings.
* **Attention Heads**: Comparison of performance between 2 and 8 attention heads.

## How to Run
1.  **Prepare Files**: Ensure `True.csv` and `Fake.csv` are in the same directory as the source files.
2.  **Run the Script**:
    ```bash
    python lastname_studentID_hw10.py
    ```
3.  **Output**: The script will print the results for all four models followed by the design study results to the console.

## Experimental Results
The models were evaluated based on accuracy, precision, recall, F1 score, mean inference speed, and parameter count.

| Model | Accuracy | F1 Score | Inf Speed (s/sample) | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **RNN** | 59.51% | 0.63 | 0.000100 | 70,273 |
| **LSTM** | 61.47% | 0.56 | 0.000117 | 89,089 |
| **Custom Transformer** | 98.26% | 0.98 | 0.000862 | 690,241 |
| **Fine-Tuned BERT** | **99.82%** | **1.00** | 0.048630 | 109,483,009 |

# 2. Model Discussion & Performance Reasoning

### Custom Transformer Implementation 
The Custom Transformer achieved a significantly high accuracy (98.26%) compared to recurrent baselines. 
* **Reason for Performance**: Unlike RNNs and LSTMs that process text sequentially—often "forgetting" earlier tokens in long news articles—the Transformer uses **Self-Attention**. This allows the model to attend to key predictive words (like "unverified" or specific political entities) regardless of their distance from each other in the text.
* **Architecture**: The inclusion of **Positional Encodings** allowed the model to maintain structural context, while the **Padding Masks** ensured that the attention mechanism did not waste computation or become biased by the zero-padding used to normalize sequence lengths.

### Fine-Tuned BERT 
BERT (Bidirectional Encoder Representations from Transformers) provided the best overall performance with an F1 score of 1.00.
* **Reason for Performance**: BERT is a **pretrained Large Language Model**. It arrives with a deep, nuanced understanding of English syntax and semantics learned from massive corpora (Wikipedia and BooksCorpus). 
* **Fine-Tuning Advantage**: While the other models had to learn "fake news" patterns and "English" simultaneously, BERT only needed to adapt its existing linguistic knowledge to the specific classification task. This results in superior precision and recall, though it comes at the cost of much slower inference speeds due to its 109 million parameters

---

## 3. Design Studies 

### Study 1: Positional Encoding (PE)
* **Observation**: Accuracy remained stable even without PE (98.17% vs 98.22%).
* **Reasoning**: This suggests that for this specific dataset, "Fake News" can often be identified by the presence of specific keywords (bag-of-words signal) rather than complex grammatical structures that require strict word-ordering.

### Study 2: Attention Heads
* **Observation**: Increasing from 2 heads to 8 heads improved accuracy from 97.77% to 98.26%.
* **Reasoning**: Multiple attention heads allow the model to focus on different aspects of the text simultaneously—such as one head focusing on subject-verb relationships while another focuses on specific punctuation patterns or sensationalist adjectives.

---
