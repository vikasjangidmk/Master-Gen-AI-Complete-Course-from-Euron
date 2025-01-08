import os
from dotenv import load_dotenv, find_dotenv
import llama_index
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# Load environment variables
load_dotenv(find_dotenv())

# Check if OPENAI_API_KEY is loaded
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables")

openai_api_key = os.environ["OPENAI_API_KEY"]
print(f"OpenAI API Key loaded successfully: {openai_api_key is not None}")

# Load private document
documents = SimpleDirectoryReader("data").load_data()

# Check if documents were loaded
if not documents:
    raise FileNotFoundError("No documents found in the 'data' folder.")

# Create vector database
index = VectorStoreIndex.from_documents(documents)

# Ask questions to private document
query_engine = index.as_query_engine()
response = query_engine.query("Summarize the article in 100 words")
print(response)

# Save the vector database
if not os.path.exists("./storage"):
    os.makedirs("./storage")
    index.storage_context.persist()  # Save the index for future use
    print("New vector store created and saved.")
else:
    # Load the existing index
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    print("Loaded existing vector store from storage.")

# Query the index again
query_engine = index.as_query_engine()
response = query_engine.query("According to the author, what is good?")
print(response)

print("The End")
