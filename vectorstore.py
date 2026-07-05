from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np

embedding_model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2'
)

def build_vectorstore(pdf_path):
    
    reader = PdfReader(pdf_path)
    
    raw_text = ""
    
    for page in reader.pages:
        raw_text += page.extract_text()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks = splitter.split_text(raw_text)
    
    embeddings = embedding_model.encode(chunks)
    
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    
    index.add(np.array(embeddings))
    
    return index, chunks
  
