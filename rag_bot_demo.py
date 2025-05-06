# pip install 
# pip install chromadb
# pip show 
# pip install langchain
# pip install tiktoken
# pip install openai langchain tiktoken chromadb python-dotenv
 
import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain.llms import OpenAI
# from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


load_dotenv()  # Make sure it replaces existing values
api_key = os.getenv("OPENAI_API_KEY") # get openAI API key
# print("Using API key:", api_key)

#Load the PDF
loader= DirectoryLoader("./docs", glob="*.pdf", loader_cls=PyPDFLoader)  # Load PDF documents from the "docs" directory
documents = loader.load()  # Load the documents into memory
print(f"Total documents loaded: {len(documents)}")
 
# Split the text
spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #Initialize a text
texts=spliter.split_documents(documents) #split loaded document into small chunk
print(f"total chunck: {len(texts)}")
print(f"first chunck: ", texts)
# print(f"first chunck: ", texts[0].page_content) # print single chunk

#create in memory  vector DB
persistant_directory = "vector_db"
embedding = OpenAIEmbeddings() #openAI embedding modal
#create directory
vectordb= Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persistant_directory)
#wrtie the chunk in drive
vectordb.persist()
vectordb=None # clear the vector db from memory

#Reload db from the persistance dir 
vectordb = Chroma(persist_directory=persistant_directory, embedding_function=embedding)

# Prepare chain settinf retrival
# retrieval = vectordb.as_retriever(search_kwargs={"k":2}) #retrive number of chunk
# llm = OpenAI()
# qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retrieval, return_source_documents=True)
# response = qa_chain.run("what are boold type?")
# print("Answer :", response["result"])

retrieval = vectordb.as_retriever(search_kwargs={"k": 4})
llm = OpenAI()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retrieval, return_source_documents=True)  # Create a retrieval-based question-answering chain
print("🤖 Chatbot: Hello! Ask me anything from the PDFs. Type 'exit' to quit.")  # Greet the user and prompt for input
while True:
    query = input("Enter your query (or type 'exit' to quit): ")  # Prompt the user for a query
    if query.lower() in ['exit', 'quit']:  # Check if the user wants to exit
        print("Exiting...")
        break  # Exit the loop if the user types 'exit' or 'quit'
    response = qa_chain(query)  # Run the QA chain with the user's query
    print("Answer:", response["result"])  # Print the answer from the QA chain