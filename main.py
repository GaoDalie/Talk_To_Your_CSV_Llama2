from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import tempfile

def main():
    st.set_page_config(page_title="👨‍💻 Talk with your CSV")
    st.title("👨‍💻 Talk with your CSV")
    st.write("Please insert your link.")
    uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

    query = st.text_input("Send a Message")
    if st.button("Submit Query", type="primary"):
        DB_FAISS_PATH = "vectorstore/db_faiss"

        if uploaded_file :
        #use tempfile because CSVLoader only accepts a file_path
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                        'delimiter': ','})
            data = loader.load()
            st.write(data)
            # Split the text into Chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
            text_chunks = text_splitter.split_documents(data)

            # Download Sentence Transformers Embedding From Hugging Face
            embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

            # COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
            docsearch = FAISS.from_documents(text_chunks, embeddings)

            docsearch.save_local(DB_FAISS_PATH)



            llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                                model_type="llama",
                                max_new_tokens=512,
                                temperature=0.1)

            qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

                # Run the query and return the result
            result = qa(query)
            st.write(result)

if __name__ == '__main__':
    main()

