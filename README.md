# **Named Entity Recognition (NER) for Swahili using RoBERTa-Base-Wechsel**

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Validation with Swahili Speakers](#validation-with-swahili-speakers)
- [Evaluation Metrics](#evaluation-metrics)
- [Results and Discussion](#results-and-discussion)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🚀 Overview
This project fine-tunes **RoBERTa-base-Wechsel-Swahili** for **Named Entity Recognition (NER) in Swahili**, using the **MasakhaNER dataset**. The goal is to identify and classify named entities such as **Persons (PER), Locations (LOC), Organizations (ORG), and Dates (DATE)** in Swahili text.  

### **Key Features:**
✅ Fine-tuning **RoBERTa-base-Wechsel-Swahili** for Swahili NER.  
✅ **Linguistic validation** with Swahili speakers for accuracy.  
✅ **Evaluation using Precision, Recall, F1-score, and Accuracy.**  
✅ **Error analysis and optimization suggestions.**  

---

## 📊 Dataset
The dataset used in this project is **MasakhaNER**, which contains Swahili text annotated with named entities.  

### **Dataset Statistics:**
| **Subset**   | **Number of Sentences** |
|-------------|-----------------------|
| **Train**   | 2,109                 |
| **Validation** | 300                 |
| **Test**     | 604                   |

Each sentence consists of:  
- `id`: Unique identifier  
- `tokens`: Words/tokens in the sentence  
- `ner_tags`: Named entity labels (`O`, `B-PER`, `I-PER`, `B-LOC`, etc.)  

### **Load the Dataset**
```python
from datasets import load_dataset
dataset = load_dataset("masakhaner", "swa", trust_remote_code=True)

Each example consists of:
- `id`: Unique identifier for the example.
- `tokens`: List of words/tokens in the sentence.
- `ner_tags`: Corresponding labels for each token (e.g., `O`, `B-PER`, `I-PER`, `B-LOC`, etc.).

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
- The tokenizer used is `AutoTokenizer` with `add_prefix_space=True` to handle tokenization of pre-split words correctly.
- The `evaluate` library is used for computing evaluation metrics.

---

## Preprocessing

The dataset is preprocessed to align the tokenized inputs with the corresponding labels. Key steps include:
1. Tokenizing the input text using the RoBERTa tokenizer.
2. Aligning the labels with the tokenized input while handling subwords.
3. Adding padding and truncation to ensure all inputs have a fixed length.

Example preprocessing function:
```python
tokenizer = AutoTokenizer.from_pretrained("benjamin/roberta-base-wechsel-swahili", add_prefix_space=True)


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
    output_dir="./results",               # Save directory
    evaluation_strategy="epoch",          # Evaluate after each epoch
    save_strategy="epoch",                # Save model after each epoch
    logging_dir="./logs",                 # Directory for logs
    logging_steps=100,                    # Log every 100 steps
    per_device_train_batch_size=16,       # Batch size for training (adjust as needed)
    per_device_eval_batch_size=32,        # Batch size for evaluation (adjust as needed)
    num_train_epochs=5,                   # Total number of epochs (adjust as needed)
    learning_rate=5e-5,                   # Learning rate (adjust as needed)
    weight_decay=0.01,                    # Weight decay for regularization
    load_best_model_at_end=True,          # Load the best model at the end
    metric_for_best_model="eval_overall_f1",  # Use eval_overall_f1 for selecting the best model
    warmup_steps=500,                     # Number of warmup steps
    logging_first_step=True,              # Log the first step
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
    model=model,                           # the model to train
    args=training_args,                    # training arguments
    data_collator=data_collator,           # data collator
    train_dataset=tokenized_datasets["train"],  # training dataset
    eval_dataset=tokenized_datasets["validation"],  # evaluation dataset
    compute_metrics=compute_metrics,       # custom metrics (precision, recall, F1)
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
    entities = []
    label_list = tokenized_test_set.features["ner_tags"].feature.names

    for pred, example in zip(predictions, tokenized_test_set):
        tokens = example["tokens"]
        word_ids = example["word_ids"]
        current_entity = []
        entity_type = None

        for idx, (p, word_id) in enumerate(zip(pred, word_ids)):
            if word_id is None:
                continue  # Skip special tokens

            label = label_list[p]
            if label != "O":  # If it's a named entity
                if not current_entity:  # Start of a new entity
                    entity_type = label.split("-")[-1]
                    current_entity.append(tokens[word_id])
                else:
                    current_entity.append(tokens[word_id])
            else:
                if current_entity:  # End of an entity
                    entities.append(("".join(current_entity), entity_type))
                    current_entity = []

        if current_entity:  # Add the last entity if it exists
            entities.append(("".join(current_entity), entity_type))

    return entities[:500]  # Return the first 500 entities
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


# Define the metric
metric = evaluate.load("seqeval")  # Using evaluate to load the seqeval metric

# Ensure that the model has a mapping for the labels
label_list = tokenized_datasets["train"].features["ner_tags"].feature.names
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {v: k for k, v in id2label.items()}

# Define data collator to handle padding
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Function to compute metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)  # Get the predicted label indices

    # Remove padding tokens (label=-100 for padding tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for l in label if l != -100]
        for label in labels
    ]

    return metric.compute(predictions=true_predictions, references=true_labels)
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

- **My University**: For the opportunity.
- **My Supervisor**: Dr. Nirav Bhatt for his guidance.
- **Hugging Face**: For providing the `transformers` and `datasets` libraries.
- **MasakhaNER**: For the annotated Swahili dataset.
- **Swahili Validators**: For their invaluable feedback during the validation phase.

---

