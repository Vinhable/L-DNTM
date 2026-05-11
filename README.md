# L-DNTM: Beyond Coherence: Improving Temporal Consistency and Interpretability in Dynamic Topic Models

Anonymous submission for dynamic topic modeling research.

## Overview

L-DNTM (LLM-Guided Neural Dynamic Topic Model) is a novel variational framework designed to capture more faithful temporal topic trajectories. The model addresses common challenges in dynamic topic modeling, such as semantic misalignment, rigid temporal linkages, and poor interpretability, by integrating three key innovations into a unified architecture:

1. Multi-Objective Distillation: Injects rich semantic priors from powerful Pre-trained Language Models (PLMs) into the document encoder to improve semantic representation.

2. Optimal Transport Regularization: Aligns the entire geometric constellation of topics between consecutive time slices, allowing for smooth yet flexible evolution that can capture complex dynamics like topic splits and merges.

3. LLM-Guided Refinement: Employs a sophisticated LLM-in-the-loop mechanism where an LLM (e.g., Google's Gemini) acts as an expert critic, providing nuanced, list-wise feedback to sharpen topic-word distributions and enhance interpretability.

## Features

- **Multi-Objective Distillation:**: Enriches the document encoder by distilling knowledge from a static, powerful Pre-trained Language Model (PLM). It uses a combination of InfoNCE loss for instance-level semantic alignment and Centered Kernel Alignment (CKA) loss to preserve the global geometric structure of the embedding space.
- **Optimal Transport for Smoothness**: Employs Wasserstein distance, calculated via the Sinkhorn algorithm, to regularize the evolution of topic embeddings, ensuring temporally consistent topic trajectories.
- **LLM-Guided Ranking Loss**: Utilizes a novel list-wise ranking loss where an LLM scores candidate words based on novelty and relevance. The model is then trained via KL-Divergence to mimic the LLM's expert ranking.
- **Hybrid Core Model**: Implements a powerful core architecture using mechanisms like Unassociated Word Exclusion (UWE) and Evolution-Tracking Contrastive learning (ETC) to enhance topic diversity and reduce noise.
- **End-to-End Reproducibility**: The main script handles everything from generating document embeddings to training and evaluation, ensuring that reviewers and users can easily replicate the results.

## Requirements

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- PyTorch 2.6.0+ (with CUDA support recommended)
- numpy==2.3.2
- pandas==2.3.1
- scipy==1.16.1
- scikit-learn==1.7.1
- gensim==4.3.3
- tqdm==4.67.1
- google-generativeai==0.8.5
- sentence-transformers==5.1.1
- topmost==1.0.2

### Set up Gemini API Key(s):

To use the LLMGuider feature, you must provide one or more Gemini API keys. The system is designed to handle multiple keys for load balancing.

Set the following environment variable, replacing the placeholders with your actual keys, separated by commas:

```python
import os
os.environ["GOOGLE_API_KEYS"] = "your_key_1,your_key_2,your_key_3"
```


## Quick Start

Simply run the main script. It is configured to run the default experiment on the NYT dataset.

```bash
python main.py
```

This will automatically:

1. Generate the required document embeddings (doc_embedding, contextual_emb) using sentence-transformers and save them to the dataset directory.

2. Train the L-DNTM model with the parameters specified in main.py.

3. Periodically query the Gemini API to guide the training process (if enabled and after the warm-up period).

4. Evaluate the final model and display a comprehensive set of metrics.

5. Save the final top words for each time slice to top_words.txt and LLM interaction logs to the ./llm_guidance_logs/ directory.

## Dataset Configuration

### Available Datasets

The model supports multiple benchmark datasets used in the paper:

- **NYT**: New York Times articles (2012-2022) - Default dataset
- **NeurIPS**: NeurIPS conference publications (1987-2017)
- **ACL**: ACL Anthology articles (1973-2006)
- **UN**: United Nations session transcripts (1970-2015)
- **WHO**: WHO articles on non-pharmacological interventions (Jan-May 2020)

### Custom Dataset

To use your own dataset, prepare it in the TopMost format and place it in the `./datasets/` directory. The dataset should contain:

- train_texts.txt, test_texts.txt: Document texts, one per line.
- train_times.txt, test_times.txt: Timestamps for each document.
- Pre-computed word embeddings (e.g., from GloVe).
- The framework will generate other necessary files like BoW representations and document embeddings.

### Running on Other Datasets (e.g., ACL, NeurIPS, UN, WHO)

The default configuration is set for the NYT dataset, which includes labels for downstream tasks. To run on other datasets that do not have labels, you need to make the following three changes in `main.py`:

1. Change the dataset directory:

```python
# Change this line
dataset_dir = "./datasets/NYT"

# To your target dataset, for example:
dataset_dir = "./datasets/ACL"
```

2. Disable label reading:

When initializing `DynamicDataset`, set `read_labels` to `False`.

```python
# Change this line
dataset = DynamicDataset(dataset_dir, batch_size=200, read_labels=True, device=device)

# To:
dataset = DynamicDataset(dataset_dir, batch_size=200, read_labels=False, device=device)
```

3. Comment out downstream task evaluation:

Since these datasets do not have labels, the clustering and classification evaluations will fail. Comment out these sections inside the `evaluate_model` function.

```python
def evaluate_model(...):
    # ... (Topic Coherence and Diversity parts are fine)

    # # Evaluate clustering -- COMMENT OUT THIS BLOCK
    # print("Evaluating clustering performance...")
    # cluster = _clustering(test_theta, dataset.test_labels)
    # purity = cluster['Purity']
    # nmi = cluster['NMI']
    # print(f"Clustering Purity: {purity:.4f}")
    # print(f"Clustering NMI: {nmi:.4f}")

    # # Evaluate classification -- COMMENT OUT THIS BLOCK
    # print("Evaluating classification performance...")
    # clf = _cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
    # acc = clf['acc']
    # f1 = clf['macro-F1']
    # print(f"Classification Accuracy: {acc:.4f}")
    # print(f"Classification F1-Score: {f1:.4f}")

    # ... (The rest of the function is fine)
```

## Training Configuration

### Basic Training

All hyperparameters are configured directly within the main.py file. The example below shows the key parameters and their roles.

```python
model = L_DNTM(
    # --- Core Model & Dataset Parameters ---
    vocab_size=dataset.vocab_size,              # Size of the vocabulary (from dataset)
    num_times=dataset.num_times,                # Number of time slices (from dataset)
    num_topics=50,                              # Number of topics to discover
    train_time_wordfreq=dataset.train_time_wordfreq.to(device), # Word frequencies per time slice (for UWE)
    word_embeddings=dataset.pretrained_WE,      # Pre-trained word embeddings (e.g., GloVe)
    en_units=200,                               # Hidden units in the main document encoder
    dropout=0.01,                               # Dropout rate in the main encoder
    beta_temp=0.7,                              # Temperature for the beta (topic-word) distribution softmax

    # --- Core Loss Component Parameters (from CFDTM) ---
    temperature=0.1,                            # Temperature for contrastive losses (ETC, UWE)
    weight_neg=7e+7,                            # Weight for ETC's negative loss (promotes diversity within a time slice)
    weight_pos=1.0,                             # Weight for ETC's positive loss (promotes smoothness between time slices)
    weight_UWE=1.0e+3,                          # Weight for Unassociated Word Exclusion loss (removes irrelevant words)
    neg_topk=15,                                # Number of top words considered by UWE

    # --- Optimal Transport Parameters ---
    ot_warm_up=150,                             # Epoch to start applying the OT loss
    weight_loss_ot=10.0,                         # Weight for the Optimal Transport loss
    ot_sinkhorn_alpha=20.0,                     # Regularization parameter for the Sinkhorn algorithm

    # --- Knowledge Distillation Parameters ---
    weight_distill_loss=5.0,                    # Overall weight for the entire distillation loss module
    distill_en_units=200,                       # Hidden units for the encoder inside the distill module
    distill_dropout=0.1,                        # Dropout for the encoder inside the distill module
    distill_w_cka=1.0,                          # Relative weight for CKA loss within the distill module
    distill_w_infonce=1.0,                      # Relative weight for InfoNCE loss within the distill module
    distill_weight_cka_internal=250.0,          # Internal gain to amplify the CKA loss signal
    distill_weight_infonce_internal=10.0,       # Internal gain to amplify the InfoNCE loss signal
    distill_infonce_proj_dim=768,               # Projection dimension for InfoNCE

    # --- LLM Guider Parameters ---
    llm_warm_up_epochs=296,                     # Epoch to start applying the LLM guidance loss
    lambda_contrastive=10.0,                    # Overall weight for the LLM's list-wise ranking loss
    gemini_model_name="gemini-2.0-flash-lite",  # Which Gemini model to use for guidance
    llm_guidance_refresh_rate=1,                # How often (in epochs) to call the LLM API
    llm_top_k=30,                               # Number of candidate words to send to the LLM for analysis
    llm_history_length=3,                       # Number of past time slices to include in the prompt
    llm_batch_size=6,                           # Number of topics to batch into a single LLM API call
    idx_to_word=dataset.idx_to_word,            # Mappings from dataset, required by LLMGuider
    word_to_idx=dataset.word_to_idx,
    llm_log_path="./llm_guidance_logs/"         # Directory to save API interaction logs
)
```


### Training Parameters

```python
trainer = DynamicTrainer(
    model,
    dataset,
    epochs=300,                  # Total number of training epochs
    learning_rate=0.002,           # Optimizer learning rate
    batch_size=200,                # Number of documents per batch
    log_interval=5,                # Frequency (in epochs) to print logs
    num_top_words=15,              # Number of final top words to display and evaluate
    verbose=True                   # Enable detailed console output
)
```

### Advanced Configuration

To modify hyperparameters for advanced experiments, edit the model initialization block within the `main.py` file. Below are some of the key parameters to tune for balancing the different components of the model:

- `num_topics`: Number of topics to discover (default: 50)
- `epochs`: Total number of training epochs (default: 300).
- `learning_rate`: Optimizer learning rate (default: 0.002)
- `weight_loss_ot`: Adjusts the influence of the Optimal Transport loss (default: 10.0)
- `weight_distill_loss`: Controls the overall strength of the knowledge distillation from the PLM. (default: 5.0)
- `lambda_contrastive`: Manages the impact of the LLM-guided ranking loss, influencing the interpretability and novelty of topics. (default: 10.0)
- `ot_warm_up`: Define the starting epoch for the OT loss components (default: 150)
- `llm_warm_up_epochs`: Define the starting epoch for the LLM loss components. (default: 296)


## Output Files

After a successful training run, the following files will be generated:

- `top_words.txt`: Contains the final top 15 words for each topic at each time slice. This list is refined by the LLM's feedback where available.
- `./llm_guidance_logs/`: This directory contains .jsonl files logging every prompt sent to the Gemini API and the raw response received. This is useful for debugging and analyzing the LLM's behavior.
- Console Output: Detailed metrics are printed to the console at the end of the run, including Topic Quality (TQ), Temporal Topic Quality (TTQ), Dynamic Topic Quality (DTQ), and more.

## GPU Support

The code will automatically use a CUDA-enabled GPU if torch.cuda.is_available() returns True. To force the model to run on the CPU, you can change the device variable at the beginning of main.py:

```python
device = 'cpu'  # Instead of 'cuda'
```
