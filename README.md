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
dataset = load_dataset("masakhaner", "swa")
