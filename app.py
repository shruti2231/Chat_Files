import joblib
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import io
import docx # type: ignore
from docx import Document # type: ignore

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    """
    This function goes through each page of each PDF file, extracts text from it, and appends it to text.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_stream = io.BytesIO(pdf.read())
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    This will get text and create chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    joblib.dump(vector_store, "faiss_index.pkl")
    #return vector_store  


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from th provided contect , make sure to provide all the details , if the answer is not in 
    provided context just say , "answer is not available in the context" , don't provide the wrong answer \n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = joblib.load("faiss_index.pkl")
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])


def read_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"  # Add a newline character after each paragraph
    return text


def read_txt(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    return text


def read_file(file):
    file_extension = file.name.split(".")[-1]

    if file_extension == "pdf":
        return get_pdf_text([file])
    elif file_extension == "docx":
        return read_docx(file)
    elif file_extension == "txt":
        return read_txt(file)
    else:
        return "Unsupported file format"


def main():
    st.set_page_config("chat with PDF's", layout="wide")
    st.header("Chat with Multiple files using Gemini")

    user_question = st.text_input("Ask a question from the files", placeholder="Enter your question here...", label_visibility="visible")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Menu")
        files = st.file_uploader("Upload your files(.pdf/.txt/.docx)", accept_multiple_files=True)
        if st.button("Submit and process"):
            if files:
                with st.spinner("Processing..."):
                    raw_text = ""
                    for file in files:
                        raw_text += read_file(file)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload at least one file.")
        st.write("Created with 💓 from Bhupendra and Shruti")
    


if __name__ == "__main__":
    main()
