import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# -------------------------
# DEVICE SETUP
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on: {device}")

# -------------------------
# LOAD MODEL
# -------------------------
model = SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2'
).to(device)

# -------------------------
# LOAD DATASET
# -------------------------
df = pd.read_csv("data/training_data.csv", header=None)

df.columns = ["question", "answer"]


questions = df["question"].astype(str).str.strip().tolist()
answers = df["answer"].astype(str).str.strip().tolist()

# -------------------------
# ENCODE QUESTIONS
# -------------------------
print("Encoding dataset...")

corpus_embeddings = model.encode(
    questions,
    convert_to_tensor=True
)

# -------------------------
# CHATBOT FUNCTION
# -------------------------
def get_answer(user_query):

    query_embedding = model.encode(
        user_query,
        convert_to_tensor=True
    )

    hits = util.semantic_search(
        query_embedding,
        corpus_embeddings,
        top_k=1
    )

    best_hit = hits[0][0]

    score = best_hit['score']
    index = best_hit['corpus_id']

    if score > 0.65:
        return answers[index], score

    return "Gaaffii keessan sirriitti hin hubanne.", score


# -------------------------
# CHAT LOOP
# -------------------------
print("\nAfaan Oromo Tourism Chatbot")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response, confidence = get_answer(user_input)

    print(f"Bot: {response}")
    print(f"Confidence: {confidence:.2f}\n")