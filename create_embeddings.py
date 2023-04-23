import openai  # for generating embeddings
import pandas as pd  # for DataFrames to store article sections and embeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

DOCUMENT_PATH = "docs/budget_speech.pdf"
EMBEDDING_EXPORT_FOLDER = "embeddings"
EXMBEDDING_EXPORT_FILE_NAME = "budget_embeddings.csv"

loader = PyPDFLoader(DOCUMENT_PATH)
pages = loader.load_and_split()

# calculate embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
BATCH_SIZE = 500  # you can submit up to 2048 embedding inputs per request
text_splitter = NLTKTextSplitter(chunk_size=1000)

embeddings = []
for batch_start in range(0, len(pages), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = [x.page_content for x in pages[batch_start:batch_end]]
    batches = []
    for t in batch:
        batches.extend(text_splitter.split_text(t))
    batch=batches
    print(f"Batch {batch_start} to {batch_end-1}")
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response["data"]):
        assert i == be["index"]  # double check embeddings are in same order as input
    batch_embeddings = [e["embedding"] for e in response["data"]]
    embeddings.extend(batch_embeddings)
    
batch = [x.page_content for x in pages]
batches = []
for t in batch:
    batches.extend(text_splitter.split_text(t))
batch=batches
df = pd.DataFrame({"text": batch, "embedding": embeddings})

# save document chunks and embeddings

SAVE_PATH = os.path.join(EMBEDDING_EXPORT_FOLDER, EXMBEDDING_EXPORT_FILE_NAME)

df.to_csv(SAVE_PATH, index=False)
