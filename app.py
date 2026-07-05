import streamlit as st
from chatbot import ask_question

st.title("📚 RAG Document Question Answering")

query = st.text_input("Ask a Question")

if st.button("Generate Answer"):
    
    answer = ask_question(query)
    
    st.write("### Answer:")
    st.write(answer)
  
