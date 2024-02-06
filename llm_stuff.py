from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import torch
import gc
def create_qa_chain():
    # Clear CUDA cache and perform garbage collection
    torch.cuda.empty_cache()
    gc.collect()

    # Initialize Ollama with necessary parameters
    llm = Ollama(
        model="mistral",
        callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
        num_gpu=1,
        base_url="http://localhost:11434"
    )
    modelPath = "BAAI/bge-large-en-v1.5"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device': 'cuda:0'}
    encode_kwargs = {'normalize_embeddings': True}
    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embedding = HuggingFaceEmbeddings(
        model_name=modelPath,  # Provide the pre-trained model's path
        model_kwargs=model_kwargs,  # Pass the model configuration options
        encode_kwargs=encode_kwargs  # Pass the encoding options
    )
    print("Embedding model loaded")

    # Load and split documents from the PDF
    loader = DirectoryLoader("./data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print("Documents loaded")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print("Documents split")

    # Create and persist a vector database
    persist_directory = './db'
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print("Vector DB created")

    # Create a retriever from the vector database
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    print("Retriever created")
    bm25_retriever = BM25Retriever.from_documents(texts)
    bm25_retriever.k =  5
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever],
                                       weights=[0.5, 0.5])
    # Create a retrieval-based QA system from the chain type
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=ensemble_retriever,
        return_source_documents=True
    )

    return qa_chain
print("QA chain created")
def process_llm_response(query, qa_chain):
    # qa_chain = create_qa_chain()
    llm_response = qa_chain(query)
    return llm_response['result']

