import warnings
import os
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initializing Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbotn"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region="us-east-1"
        )
    )

# Get the index object
index = pc.Index(index_name)

# Check if index is empty
stats = index.describe_index_stats()
record_count = stats.get('total_vector_count', 0)

if record_count == 0:
    # If index is empty, add embeddings
    docsearch = PineconeVectorStore.from_texts(
        [t.page_content for t in text_chunks], 
        embeddings, 
        index_name=index_name,
        pinecone_api_key=PINECONE_API_KEY
    )
else:
    # If index has records, just connect to it
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    

