from transformers import pipeline
from vectorstore import build_vectorstore
from sentence_transformers import SentenceTransformer
import numpy as np

embedding_model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2'
)

index, chunks = build_vectorstore("sample.pdf")

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
)

def ask_question(query):
    
    query_embedding = embedding_model.encode([query])
    
    distances, indices = index.search(
        np.array(query_embedding),
        3
    )
    
    retrieved = []
    
    for idx in indices[0]:
        retrieved.append(chunks[idx])
    
    context = "\n".join(retrieved)
    
    prompt = f"""
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """
    
    response = llm(prompt)
    
    return response[0]['generated_text']
