#   Hallucination Detector In Answers

A lightweight, advanced Python tool to detect hallucinations in generated answers. It uses a local **RoBERTa-based Natural Language Inference (NLI)** model. It does not require any paid API keys and works completely offline.

_________________________________________________________________________________________________________________________________________________________________________________________________________________________________

## 🚀 Key Features

* **No API Key Required**: Runs entirely on your local machine for free.
* **Semantic Analysis**: Understands the meaning of words instead of just looking for exact keyword matches.
* **Structured Output**: Generates clean, ready-to-use JSON results with confidence scores.

_________________________________________________________________________________________________________________________________________________________________________________________________________________________________


## 📊 Output Format

The tool compares a generated answer against its source context and outputs a clean JSON response like this:

```json
{
  "label": "hallucinated",
  "confidence": 0.85
}
```

_________________________________________________________________________________________________________________________________________________________________________________________________________________________________


## 🛠️ Requirements & Installation

Make sure Python is installed on your computer. Then, install the required packages using your terminal:

```bash
pip install torch transformers
```

_________________________________________________________________________________________________________________________________________________________________________________________________________________________________


## 💻 How to Run It

1. Clone or download this repository.
2. Save the code to a file named `detector.py`.
3. Run the script using your terminal:

```bash
python detector.py
```
_________________________________________________________________________________________________________________________________________________________________________________________________________________________________

### Quick Code Example

```python
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model locally
model_name = "cross-encoder/nli-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Your context and answer text
context = "France is a country in Western Europe. Its capital city is Paris."
answer = "The capital of France is Berlin, which is known for its nightlife."

# Run detection logic here (see detector.py for full code)
```

_________________________________________________________________________________________________________________________________________________________________________________________________________________________________


## 🧪 How It Works

This project treats hallucination detection as a **Natural Language Inference (NLI)** problem [source: 1]:
1. **Context** acts as the source truth (Premise) .
2. **Answer** acts as the text to check (Hypothesis) .
3. The model classifies the text:
   * **Contradiction** = `hallucinated` ❌ 
   * **Entailment** = `grounded` ✅ 
   * **Neutral** = `neutral` 🤷

_________________________________________________________________________________________________________________________________________________________________________________________________________________________________


## 👤 Author & Contact

* **Name**: Mehak Sharma
* **Email**: mehak.ssrr@gmail.com
* **GitHub**: https://github.com/mehakssrr

____________________________________________________________________________________________________________________________________________________________________________________________________
