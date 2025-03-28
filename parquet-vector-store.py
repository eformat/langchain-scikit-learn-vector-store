from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import tempfile
import os

# Load a sample document corpus
loader = TextLoader("paul_graham_essay.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings()

# Create the SKLearnVectorStore, index the document corpus and run a sample query
persist_path = os.path.join(tempfile.gettempdir(), "union.parquet")

vector_store = SKLearnVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_path=persist_path,  # persist_path and serializer are optional
    serializer="parquet",
)

query = "How much seed funding did Viaweb get?"
docs = vector_store.similarity_search(query)

print(f"Q: ", query)
print(f"A: ", docs[0].page_content)
print("\n")

# Saving and loading a vector store
vector_store.persist()
print(">> Vector store was persisted to", persist_path)
print("\n")

vector_store2 = SKLearnVectorStore(
    embedding=embeddings, persist_path=persist_path, serializer="parquet"
)
print(">> A new instance of vector store was loaded from", persist_path)
print("\n")

docs = vector_store2.similarity_search(query)
print(f"Q: ", query)
print(f"A: ", docs[0].page_content)

# Cleanup
#os.remove(persist_path)
