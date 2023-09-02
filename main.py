import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import CTransformers
import os
import time

def main():
    os.environ['HF_API_TOKEN'] = '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'
    st.set_page_config(page_title="Ask your PDF")
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
        progress_bar = st.progress(0)
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            progress_bar.progress(25)

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        progress_bar.progress(50)

        # create embeddings
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cpu'})
                                       
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        progress_bar.progress(75)

        # show user input
        user_question = st.text_input("Ask a question about your PDF:", key="pdf_question")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = CTransformers(
                model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
                model_type="llama",
                max_new_tokens = 128,
                temperature = 0.5
            )
            #chain = load_qa_chain(llm, chain_type="stuff")
            #response = chain.run(input_documents=docs, question=user_question)
            
        progress_bar.progress(100)
        
        time.sleep(0.5)
        progress_bar.empty()
        
if __name__ == '__main__':
    main()
