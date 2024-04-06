import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import FAISS ## vector embeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import requests



load_dotenv()
# os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):## return text from multiple pdf docs
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf) ## list of pages
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text




## chunkify the text 
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


## get vector embeddings  from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings) ## vector embeddings generated 
    vector_store.save_local("faiss_index") ## save vector embeddings 


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible in a pointwise manner from the provided context, make sure to provide maximum details,
    make sure to keep the answer as close to the given context as possible.
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.4)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def gptmagic(input):
    URL = "https://api.openai.com/v1/chat/completions"
    OPEN_API_KEY = "" ## add key here

    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": f"Make this answer into an answer that i would write in an examination , make it as detailed and long as poissible , keep it pointwise.\n \n The Answer:{input}"}],
    "temperature" : 1.0,
    "top_p":1.0,
    "n" : 1,
    "stream": False,
    "presence_penalty":0,
    "frequency_penalty":0,
    }

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPEN_API_KEY}"
    }

    response = requests.post(URL, headers=headers, json=payload, stream=False)
    print(response.content)

    # print(response.content)
    response_json = response.json()
    
    # # Extract the "content" part
    content = response_json['choices'][0]['message']['content']

    
    # Now 'content' variable holds the extracted content
    return content





def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,  allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response["output_text"])
    ## gpt enhancement here ( requires open ai paid)
    # out = gptmagic(response["output_text"])

    st.write("Reply: ", response["output_text"])





def main():
    st.set_page_config("OneNightCram")
    st.header("Lets Cram!")

    user_question = st.text_input("Ask Question From the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Material and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):## creating FAISS index from input file
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()