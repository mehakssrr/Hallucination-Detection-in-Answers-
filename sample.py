#general code for hallucination ( the output mentioned in the task is not being presented here only a determined value is being taken)

pip install pandas scikit-learn
import io
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Loaded the dataset from the text block
csv_data = """id,question,context,answer,expected_label,category,difficulty,notes
T001,What is the capital of France?,"France is a country in Western Europe. Its capital and largest city is Paris, which has been the country's capital since the 10th century.","The capital of France is Paris.",relevant,factual_qa,easy,Clear grounded answer
T002,What is the capital of France?,"France is a country in Western Europe. Its capital and largest city is Paris, which has been the country's capital since the 10th century.","The capital of France is Berlin, which is known for its nightlife.",hallucinated,factual_qa,easy,Direct contradiction of context
T003,What is the capital of France?,"France is a country in Western Europe. Its capital and largest city is Paris, which has been the country's capital since the 10th century.","France has a rich culture and history.",irrelevant,factual_qa,easy,Avoids the question entirely
T004,How does photosynthesis work?,"Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce oxygen and glucose.","Plants use sunlight, water, and CO2 to produce glucose and oxygen through photosynthesis.",relevant,science,easy,Paraphrase of context — still correct
T005,How does photosynthesis work?,"Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce oxygen and glucose.","Photosynthesis is when plants absorb moonlight and convert it into nitrogen gas.",hallucinated,science,easy,Fabricated facts not in context
"""

# Loaded into a data table
df = pd.read_csv(io.StringIO(csv_data))

# 2. Created a text-to-number tool
vectorizer = TfidfVectorizer()


# 3. Created a helper function to check text matching:

def check_hallucination(row):
    # If there is no context text, we cannot check it
    if pd.isna(row["context"]) or row["context"] == "":
        return "Cannot check (No Context)"

    context = str(row["context"])
    answer = str(row["answer"])

    # Converted the texts into numbers
    tfidf_matrix = vectorizer.fit_transform([context, answer])

    # Calculated the similarity score (0.0 to 1.0)
    #changing the scores on random.
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Set a rule: if match score is below 0.35, flag it as a hallucination
    # rules can be changed if 0.35 seems to be small take a bigger value
    if score >= 0.35:
        return "Grounded (Safe)"
    else:
        return "Hallucinated (Flagged)"


# 4. Run the check on every row
df["detected_label"] = df.apply(check_hallucination, axis=1)

# Display our findings
print(df[["id", "expected_label", "detected_label"]])
