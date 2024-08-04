
import streamlit as st
from PyPDF2 import PdfReader  
from langchain.text_splitter import RecursiveCharacterTextSplitter  # https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings # 

import google.generativeai as genai
from langchain.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # take environment variables from .env.


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):  
    """
    This function extracts text from multiple PDF documents.
    Args:
        pdf_docs: A list of file paths to the PDF documents.
    Returns:
        A single string containing all the extracted text.
    """

    # Initialize an empty string to store the extracted text
    text = "" 

    # Iterate through each PDF file path in the list
    for pdf in pdf_docs:  
        
        # Open the PDF file using PdfReader (assumes you have a suitable library installed)
        pdf_reader = PdfReader(pdf)  

        # Iterate through each page in the PDF
        for page in pdf_reader.pages:  
            
            # Extract the text from the current page and append it to the 'text' variable
            text += page.extract_text()  

    # Return the   combined text from all PDFs
    return text  

def get_text_chunks(text):
    """
    This function splits the input text into smaller chunks for efficient processing.

    Args:
        text: The input text to be split into chunks.

    Returns:
        A list of text chunks.
    """

    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    # Split the input text into chunks and return the list of chunks
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings =GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    """
    This function creates a vector store from the input text chunks.

    Args:
        text_chunks: A list of text chunks.

    Returns:
        A vector store object.
    """

# Create a FAISS vector store from the text chunks ie Facebooks vector store
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
# Explanation:
# This line creates a FAISS index (a type of vector store) from the text chunks. 
# FAISS is an efficient library for similarity search and clustering of dense vectors.
# `text_chunks`: This should be a list of strings representing the text you want to index. Each string is likely a chunk of a larger document.
# `embedding`: This is the embedding model object you created earlier (e.g., `GoogleGenerativeAIEmbeddings`). It will be used to convert each text chunk into a vector representation.
# The `FAISS.from_texts` function automatically generates embeddings for your text chunks using the provided embedding model. It then builds an index structure to enable efficient similarity search on those embeddings.

# Save the vector store locally
    vector_store.save_local('faiss_index')
# Explanation:
# This line saves the created FAISS index to your local file system in a directory named 'faiss_index'. 
# This allows you to persist the index so you don't have to rebuild it from scratch every time you run your application.

def get_conversational_chain():
    """
    This function sets up a conversational chain using a pre-defined prompt template.
    Returns:
        A conversational chain object.
    """
# Define the prompt template for the conversational chain
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide 
        all the details, if the answer is not in provided context just say, "answer is not available in context", 
         don't provide the wrong answer \n\n
      Context \n{context}\?n\
      Question:\n {question}\n

      Answer:     
    """    
# Explanation:
# This sets up the instructions that will guide the language model's responses.
# It emphasizes detailed answers, the importance of using the provided context, and avoiding incorrect responses when information isn't available.
# The placeholders {context} and {question} will be filled in later with actual content.
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
# Explanation:
# This initializes a language model from Google's Generative AI suite. 
# "gemini-pro" is likely a specific model name, chosen for its capabilities.
# The temperature setting of 0.3 makes the model's output more focused and deterministic, avoiding overly creative responses.
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# Explanation:
# This creates a structured prompt template. 
# It combines the instructions (`prompt_template`) with the variable names to be filled in during the conversation. 
# This ensures that the model receives input in a consistent format.
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
# Explanation:
# This constructs a Question Answering (QA) chain. It's designed to take a question and context as input, then generate a relevant answer. 
# The "stuff" chain_type likely means it's a simple chain that stuffs all the context into the prompt at once.
# It uses the specified language model (`model`) and the formatted prompt template (`prompt_template`) to structure its interactions.
    return chain

# Explanation:
# This line makes the created QA chain available for use elsewhere in your code.
# You can now feed it questions and context, and it will use the language model to generate answers based on the provided information.


def user_input(user_question):
    # Load the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Explanation:
    # This line initializes an object `embeddings` from the `ChatGoogleGenerativeAIEmbeddings` class.
    # It specifies that the "embedding-001" model should be used.
    # This model is responsible for converting text into numerical representations (embeddings) that capture semantic meaning.

    # Load the FAISS index
    new_db = FAISS.load_local('faiss_index', embeddings)

    # Explanation:
    # This line loads a FAISS index from the local file system.
    # FAISS (Facebook AI Similarity Search) is a library designed for efficient similarity search and clustering of dense vectors.
    # The index was  created previously (e.g., by indexing documents or text chunks) and saved to the 'faiss_index' directory.
    # The `embeddings` object is passed here because the index is  structured to work with embeddings generated by this specific model.

    # Perform similarity search
    docs = new_db.similarity_search(user_question)

    # Explanation:
    # This line uses the loaded FAISS index to perform a similarity search.
    # The `user_question` is first converted into an embedding using the `embeddings` model.
    # Then, the FAISS index is searched for embeddings that are most similar to the query embedding.
    # The result is a list of documents (or text chunks) that are deemed most relevant to the `user_question`.

    # Get the conversational chain
    chain = get_conversational_chain()

    # Explanation:
    # This line assumes that you have a function `get_conversational_chain()` defined elsewhere in your code.
    # This function creates a conversational chain, which is a sequence of components that can be used to generate responses to user inputs.
    # The chain may include components like language models, prompt templates, and other tools for processing text.

    # Generate a response using the chain
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Explanation:
    # This line sends the retrieved documents (`docs`) and the original `user_question` as input to the `chain`.
    # The `chain` processes this input and generates a response, which is stored in the `response` variable.
    # `return_only_outputs=True` indicates that you only want the text output of the chain, not any other intermediate results.

    # Display the response
    st.write("Reply:", response["output_text"])

    # Explanation:
    # `st` refers to a library like Streamlit, this line displays the generated response (`response["output_text"]`) in a user interface.

# Creaeting streamlit app
def main():
    st.set_page_config= st.text_input("Ask a Question from about LLMs?")
    st.header("Chat with top 10 LLM papers availalble today using Gemini")
     
    user_question = st.text_input("Ask a Question about LLMs? ")

    if user_question:
        user_input(user_question)

    with st.sidebar:  # this allows us to upload pdfs and turn into vectors 
        st.title("Menu:")
        pdf_docs =st.file_uploader("Upload any PDFs papers you want to add and Click the Submit Button")
        if st.button("Submit"):
            if pdf_docs is not None:
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")