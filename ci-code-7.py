'''
Collective Intelligence with State-Machine and Dream Injector.

Key Enhancements:
- Thread Control with stop_event: The dream_state_thread function now uses a stop_event to allow controlled stopping of the thread.
- File Saving in Dream State: The generated dream state content is saved to a file with a timestamp in the filename.
- Thread Synchronization: The thread is started and then joined, ensuring that the main code waits for the dream state thread to complete before proceeding.
- Error Handling for Dream Command: Added a try-except block to handle cases where the dream command is not formatted correctly.

Note:
vectorDataOne and vectorDataTwo contains the seeds of this multi-demensional VectorDB and multi-LLM with dream injector [LLM] 
vectorDataThree contains the results of the LLM response
vectorDataFour contains the dream results
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
import subprocess

PORT = 11434
warnings.filterwarnings("ignore")

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

def dream_state_thread(faiss_retriever, llm, stop_event, duration=60):
    start_time = time.time()
    while not stop_event.is_set() and time.time() - start_time < duration:
        dream_state_content = generate_dream_state(faiss_retriever, llm)
        print("\nDream State Content Generated:\n", dream_state_content)
        
        timestamp = int(time.time())
        filename = f"vectorDataFour/dream_result_{timestamp}.txt"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"The DREAM Date and Time: {current_time}\n\n{dream_state_content}")
        
        time.sleep(5)

    print("Exiting dream state...")

def check_port(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python faiss-llm-feedback.py <llm model>")
        sys.exit(1)

    llm_model = sys.argv[1]
    print(f"\nReceived LLM Model: {llm_model}")

    print("Processing vectorDataOne...")
    vectorDataOne_index = "vectorDataOne_index"
    faiss_vectorstore_one = process_vector_data("./vectorDataOne", vectorDataOne_index)

    print("Processing vectorDataTwo...")
    vectorDataTwo_index = "vectorDataTwo_index"
    faiss_vectorstore_two = process_vector_data("./vectorDataTwo", vectorDataTwo_index)

    print("Processing vectorDataThree...")
    vectorDataThree_index = "vectorDataThree_index"
    faiss_vectorstore_three = process_vector_data("./vectorDataThree", vectorDataThree_index)

    print("Processing vectorDataFour...")
    vectorDataFour_index = "vectorDataFour_index"
    faiss_vectorstore_four = process_vector_data("./vectorDataFour", vectorDataFour_index)

    faiss_vectorstore_one.merge_from(faiss_vectorstore_two)
    faiss_vectorstore_one.merge_from(faiss_vectorstore_three)
    faiss_vectorstore_one.merge_from(faiss_vectorstore_four)

    merged_index_name = "merged_vectorData_index"
    faiss_vectorstore_one.save_local(merged_index_name)

    faiss_retriever = faiss_vectorstore_one.as_retriever(search_kwargs={"k": 5})

    template = """Use the following pieces of context to answer the question at the end.
    If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    llm = Ollama(model=llm_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=faiss_retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    ##faiss_retriever_four = faiss_vectorstore_four.as_retriever(search_kwargs={"k": 5})

    ##QA_CHAIN_PROMPT_DREAM = PromptTemplate(input_variables=["context", "question"], template=template)

    ####dream_llm = Ollama(model=llm_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    ##dream_llm = Ollama(model=llm_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=3)

    ##dream_qa_chain = RetrievalQA.from_chain_type(
    ##    llm=dream_llm,
    ##    retriever=faiss_retriever_four,
    ##    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT_DREAM},
    ##)

    if not check_port(PORT):
        print(f"\nPort {PORT} is not open. Please start the LLM engine.")
        sys.exit(1)

    print(f"\n\nCheck for readme in vectorData folders.")
    print(f"\nPrompt suggestion 1:\nwhat is this story about?")
    print(f"\nPrompt suggestion 2:\ntell me about the current dreams in the matrix.")

    while True:
        if not check_port(PORT):
            print(f"\nPort {PORT} is not open. Exiting...")
            print(f"\nPlease start the LLM engine on port {PORT}.")
            sys.exit(1)

        query = input("\n\nEnter your query (type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting...")
            break
        elif query.lower().startswith("dream"):
            try:
                parts = query.split()
                seconds = int(parts[1])
                model_name = parts[2] if len(parts) > 2 else llm_model

                # Launch dream_launch.py in a separate process
                subprocess.Popen([sys.executable, "dream_launch.py", str(seconds), model_name])

                # Initialize the dream LLM with the specified model
                ##dream_llm = Ollama(model=model_name, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=3)

                ##stop_event = threading.Event()
                ##dream_thread = threading.Thread(target=dream_state_thread, args=(faiss_retriever_four, dream_llm, stop_event, seconds))
                ##dream_thread.start()
                
                ##dream_thread.join()
                ##stop_event.set()
            except ValueError:
                print("\nInvalid format for 'dream' command. Use: dream <seconds>")

        else:
            result = qa_chain({"query": query})

            timestamp = int(time.time())
            filename = f"vectorDataThree/result_{timestamp}.txt"
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(f"Date and Time: {current_time}\n\n{str(result)}")

            faiss_vectorstore_one = load_new_data(str(result), faiss_vectorstore_one, merged_index_name)

            faiss_retriever = faiss_vectorstore_one.as_retriever(search_kwargs={"k": 5})

            qa_chain.retriever = faiss_retriever



