from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from langchain.vectorstores import Pinecone

from config import PINECONE_API_ENV,PINECONE_API_KEY,EMBEDDING_MODEL_PATH,INDEX_NAME
#Extract data from the PDF

class PDFTextExtractor:
    def __init__(self, folder_path):
        self.folder_path = folder_path


    def load_pdf(self,data):
        loader = DirectoryLoader(data,
                        glob="*.pdf",
                        loader_cls=PyPDFLoader)
        
        documents = loader.load()

        return documents

    def text_split(self,extracted_data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
        text_chunks = text_splitter.split_documents(extracted_data)

        return text_chunks
    def extract_text(self):
        extracted_data = self.load_pdf(self.folder_path)
        text_chunks = self.text_split(extracted_data)
        return text_chunks

class PineconeVectorDatabase:
    def __init__(self,text_chunks):
        self.api_key=PINECONE_API_KEY
        self.environment=PINECONE_API_ENV
        self.text_chunks = text_chunks

    def store_vector(self):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)


        #Initializing the Pinecone
        pinecone.init(api_key=PINECONE_API_KEY,
                    environment=PINECONE_API_ENV)
        if INDEX_NAME not in pinecone.list_indexes():
        # we create a new index
            pinecone.create_index(name=INDEX_NAME, metric="cosine", dimension=384)

        #Creating Embeddings for Each of The Text Chunks & storing
        docsearch=Pinecone.from_texts([t.page_content for t in self.text_chunks], embeddings,metadatas=[{'pdf_name': p.metadata["source"].split("\\")[-1].lower()} for p in self.text_chunks], index_name=INDEX_NAME)