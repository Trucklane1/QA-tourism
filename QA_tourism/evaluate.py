import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# -------------------------
# DEVICE
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# MODEL
# -------------------------
model = SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2'
).to(device)

# -------------------------
# LOAD TRAIN DATA
# -------------------------
df_train = pd.read_csv(
    "data/training_data.csv",
    header=None
)

df_train.columns = ["question", "answer"]

train_questions = df_train["question"].tolist()
train_answers = df_train["answer"].tolist()

# -------------------------
# LOAD TEST DATA
# -------------------------
df_test = pd.read_csv(
    "data/testing_data.csv",
    header=None
)

df_test.columns = ["question", "answer"]

test_questions = df_test["question"].tolist()
test_answers = df_test["answer"].tolist()

# -------------------------
# ENCODE TRAIN QUESTIONS
# -------------------------
print("Encoding training data...")

corpus_embeddings = model.encode(
    train_questions,
    convert_to_tensor=True
)

# Precompute expected answer embeddings
test_answer_embeddings = model.encode(
    test_answers,
    convert_to_tensor=True
)

# -------------------------
# EVALUATION
# -------------------------
correct_count = 0
threshold = 0.75

for i in range(len(test_questions)):

    query = test_questions[i]
    expected_emb = test_answer_embeddings[i]

    query_emb = model.encode(
        query,
        convert_to_tensor=True
    )

    hits = util.semantic_search(
        query_emb,
        corpus_embeddings,
        top_k=1
    )

    idx = hits[0][0]['corpus_id']

    predicted_answer = train_answers[idx]

    predicted_emb = model.encode(
        predicted_answer,
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        predicted_emb,
        expected_emb
    ).item()

    if similarity > threshold:
        correct_count += 1

    else:
        print("\n Mismatch")
        print("Query:", query)
        print("Expected:", test_answers[i])
        print("Predicted:", predicted_answer)
        print(f"Similarity: {similarity:.2f}")

# -------------------------
# FINAL ACCURACY
# -------------------------
accuracy = (
    correct_count / len(test_questions)
) * 100

print("\n" + "-" * 30)
print(f"FINAL ACCURACY: {accuracy:.2f}%")