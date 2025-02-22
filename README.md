---

# **Build Models to Perform Name-Entity Recognization in Swahili**

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Environment Setup](#environment-setup)
4. [Preprocessing](#preprocessing)
5. [Model Training](#model-training)
6. [Validation with Swahili Speakers](#validation-with-swahili-speakers)
7. [Evaluation Metrics](#evaluation-metrics)
8. [How to Use](#how-to-use)
9. [Contributing](#contributing)
10. [License](#license)

---

## Overview

This project focuses on fine-tuning a pre-trained RoBERTa model (`benjamin/roberta-base-wechsel-swahili`) for Named Entity Recognition (NER) in Swahili using the MasakhaNER dataset. The goal is to identify and classify named entities such as PERSON, LOCATION, ORGANIZATION, DATE, etc., in Swahili text.

The project includes:
- Fine-tuning the model for Swahili NER.
- Validating predictions with Swahili speakers for linguistic accuracy.
- Evaluating performance using metrics like Precision, Recall, F1-Score, and Accuracy.

---

## Dataset

The dataset used in this project is the **MasakhaNER** dataset, which contains Swahili text annotated with named entities. It is split into three subsets:
- **Train**: 2,109 examples
- **Validation**: 300 examples
- **Test**: 604 examples

Each example consists of:
- `id`: Unique identifier for the example.
- `tokens`: List of words/tokens in the sentence.
- `ner_tags`: Corresponding labels for each token (e.g., `O`, `B-PER`, `I-PER`, `B-LOC`, etc.).

You can load the dataset using the `datasets` library:
```python
from datasets import load_dataset
dataset = load_dataset("masakhaner", "swa")
```

---

## Environment Setup

To run this project, ensure you have the following dependencies installed:

### Required Libraries
```bash
pip install transformers datasets seqeval evaluate
```

### Python Version
This project uses Python 3.11. Ensure your environment matches this version.

### Additional Notes
- The tokenizer used is `RobertaTokenizerFast` with `add_prefix_space=True` to handle tokenization of pre-split words correctly.
- The `evaluate` library is used for computing evaluation metrics.

---

## Preprocessing

The dataset is preprocessed to align the tokenized inputs with the corresponding labels. Key steps include:
1. Tokenizing the input text using the RoBERTa tokenizer.
2. Aligning the labels with the tokenized input while handling subwords.
3. Adding padding and truncation to ensure all inputs have a fixed length.

Example preprocessing function:
```python
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=128,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # New word
            else:
                label_ids.append(-100)  # Subword of the same word
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
```

---

## Model Training

The model is fine-tuned using the `Trainer` API from the Hugging Face `transformers` library. Below are the key training parameters:

### Training Arguments
```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)
```

### Data Collator
A `DataCollatorForTokenClassification` is used to handle padding and truncation during training:
```python
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

### Trainer Initialization
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
```

### Training
```python
trainer.train()
```

---

## Validation with Swahili Speakers

After training, the model's predictions are validated by Swahili speakers to ensure linguistic accuracy. The process involves:
1. Extracting 500 named entities from the test set predictions.
2. Creating a CSV file containing the extracted entities for human validation.
3. Collecting feedback from validators and calculating linguistic accuracy.

### Example Validation Code
```python
def extract_named_entities(predictions, tokenized_test_set):
    entity_count = 0
    sample_entities = []
    for i, prediction in enumerate(predictions):
        tokenized_input = tokenized_test_set[i]
        word_ids = tokenized_input.get("word_ids", [])
        tokens = tokenized_test_set["tokens"][i]

        current_entity = None
        entity_type = None
        for j, (p, word_id) in enumerate(zip(prediction, word_ids)):
            if word_id is None or word_id >= len(tokens):
                continue

            if p != 0 and p % 2 == 1:  # Start of an entity (B- prefix)
                if current_entity:
                    sample_entities.append((current_entity, entity_type))
                    entity_count += 1
                    if entity_count >= 500:
                        break
                current_entity = [tokens[word_id]]
                entity_type = dataset["train"].features["ner_tags"].feature.names[p // 2]

            elif p != 0 and p % 2 == 0:  # Inside the same entity (I- prefix)
                if current_entity:
                    current_entity.append(tokens[word_id])

            else:  # End of an entity
                if current_entity:
                    sample_entities.append((current_entity, entity_type))
                    entity_count += 1
                    if entity_count >= 500:
                        break
                current_entity = None

        if entity_count >= 500:
            break

    return sample_entities[:500]
```

---

## Evaluation Metrics

The model's performance is evaluated using the `seqeval` library, which computes the following metrics:
- **Precision**
- **Recall**
- **F1-Score**
- **Accuracy**

Example evaluation code:
```python
import numpy as np
import evaluate

metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [dataset["train"].features["ner_tags"].feature.names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [dataset["train"].features["ner_tags"].feature.names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
```

---

## How to Use

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/masakhane-ner.git
   cd masakhane-ner
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Load and Preprocess the Dataset**
   Run the preprocessing script to tokenize and align the dataset:
   ```bash
   python preprocess.py
   ```

4. **Train the Model**
   Start the fine-tuning process:
   ```bash
   python train.py
   ```

5. **Validate Predictions**
   Extract named entities and validate them with Swahili speakers:
   ```bash
   python validate.py
   ```

6. **Evaluate Performance**
   Compute evaluation metrics:
   ```bash
   python evaluate.py
   ```

---

## Contributing

We welcome contributions to improve this project! Here's how you can contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure all tests pass.
4. Submit a pull request with a clear description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Hugging Face**: For providing the `transformers` and `datasets` libraries.
- **MasakhaNER**: For the annotated Swahili dataset.
- **Swahili Validators**: For their invaluable feedback during the validation phase.

---

