'''
Summary: dream_launch.py: This script is responsible for running the dream state logic independently. It receives the duration and model name as command-line arguments.
python dream_launch.py <seconds in dream state> <llm> 

TODO:
To modify the dream_launch.py script to output to a designated port (9999)
python dream_launch.py <seconds in dream state> <llm> <port>

'''
import os
import sys
import pdfplumber
import time  # For generating Unix timestamp
from datetime import datetime  # For getting the current date and time
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from port_checker import check_port, check_curl_response
import warnings
import threading  # For creating threads
import socket

warnings.filterwarnings("ignore")

HOST = 'localhost'
PORT = 8889

class Document:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}

def load_text_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(Document(content))
        elif filename.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    content = page.extract_text()
                    if content:
                        documents.append(Document(content))
    return documents

def process_vector_data(directory_path, index_name):
    text_documents = load_text_documents(directory_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(text_documents)
    hugg_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    faiss_vectorstore = FAISS.from_documents(documents=all_splits, embedding=hugg_embeddings)
    faiss_vectorstore.save_local(index_name)
    return faiss_vectorstore

def load_new_data(text_data, faiss_vectorstore, index_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents([Document(text_data)])
    hugg_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    faiss_vectorstore.add_documents(documents=all_splits, embedding=hugg_embeddings)
    faiss_vectorstore.save_local(index_name)
    return faiss_vectorstore

def generate_dream_state(faiss_retriever, llm):
    dream_prompt = "Select a random pivotal moment from this vectorDB ONLY and create an imaginative fictional story around it."
    dream_result = llm(dream_prompt)
    return dream_result

def send_data_to_client(client_socket, data):
    try:
        client_socket.sendall(data.encode('utf-8'))
    except Exception as e:
        print(f"Error sending data: {e}")

def dream_state_thread(faiss_retriever, llm, stop_event, duration=60, client_socket=None):
    start_time = time.time()
    while not stop_event.is_set() and time.time() - start_time < duration:
        dream_state_content = generate_dream_state(faiss_retriever, llm)
        timestamp = int(time.time())
        filename = f"vectorDataFour/dream_result_{timestamp}.txt"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"The DREAM Date and Time: {current_time}\n\n{dream_state_content}")

        # Send the dream state content to the client
        if client_socket:
            send_data_to_client(client_socket, f"\nDream State Content Generated:\n{dream_state_content}\n")

        time.sleep(5)

    print("Exiting dream state...")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dream_launch.py <seconds in the dream state> <llm model>")
        sys.exit(1)

    duration = int(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'llama3.1'

    # Initialize the dream LLM with the specified model
    faiss_retriever_four = None  

    print("Launching Dream Thread...")
    vectorDataFour_index = "vectorDataFour_index"
    faiss_vectorstore_four = process_vector_data("./vectorDataFour", vectorDataFour_index)

    faiss_retriever_four = faiss_vectorstore_four.as_retriever(search_kwargs={"k": 5})

    template = """Use the following pieces of context to answer the question at the end.
    If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT_DREAM = PromptTemplate(input_variables=["context", "question"], template=template)

    dream_llm = Ollama(model=model_name, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=3)

    dream_qa_chain = RetrievalQA.from_chain_type(
        llm=dream_llm,
        retriever=faiss_retriever_four,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT_DREAM},
    )

    # Set up server to listen on the specified port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Server listening on {HOST}:{PORT}...")

        client_socket, client_address = server_socket.accept()
        with client_socket:
            print(f"Connected by {client_address}")

            stop_event = threading.Event()
            dream_thread = threading.Thread(target=dream_state_thread, args=(faiss_retriever_four, dream_llm, stop_event, duration, client_socket))
            dream_thread.start()

            dream_thread.join()
            stop_event.set()

            print("Closing connection...<press enter to exit>")
