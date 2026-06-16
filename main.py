import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Loaded the pre-trained NLI model locally (Free)

model_name = "cross-encoder/nli-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# 2. Optimized function to output exactly what was given in the task.

def detect_hallucination(context, answer):
    # Packed the context and answer together for RoBERTa

    inputs = tokenizer(
        context, answer, return_tensors="pt", truncation=True, padding=True
    )

    # Get raw model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Converted raw outputs to probabilities (percentages are taken between 0 and 1)
    probs = torch.softmax(logits, dim=1).squeeze().tolist()

    # The NLI model output positions are:
    # Index 0 = Contradiction (Hallucinated)
    # Index 1 = Entailment (Grounded)
    # Index 2 = Neutral
    contradiction_prob = probs[0]
    entailment_prob = probs[1]

    # Decided the final label 
    # set the confidence score
    if contradiction_prob > 0.5:
        final_label = "hallucinated"
        confidence = contradiction_prob
    elif entailment_prob > 0.4:
        final_label = "grounded"
        confidence = entailment_prob
    else:
        final_label = "neutral"
        confidence = probs[2]

    # Created the exact structured dictionary which was required for the task output
    output_dict = {"label": final_label, "confidence": round(confidence, 2)}

    # Converted the dictionary into a formatted JSON string

    return json.dumps(output_dict, indent=2)


#testing the model
sample_context = (
    "France is a country in Western Europe. Its capital city is Paris."
)
sample_answer = (
    "The capital of France is Berlin, which is known for its nightlife."
)

# Run the detection
final_output = detect_hallucination(sample_context, sample_answer)
print(f"Output:\n{final_output}")
