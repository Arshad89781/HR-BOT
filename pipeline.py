from data_extraction import PDFTextExtractor,PineconeVectorDatabase
from model_response import LLMResponse

class ExtractionAndVectorStoringTask:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def vector_storing(self):
        text_extractor = PDFTextExtractor(folder_path=self.folder_path)
        VectorData = PineconeVectorDatabase()
        VectorData.store_vector(text_chunks=text_extractor.extract_text())

class ResponseTask:
    def __init__(self, query):
        self.query = query
    def get_response(self):
        VectorData = PineconeVectorDatabase()
        doc_vectors = VectorData.fetch_vectors()
        response = LLMResponse(query=self.query,docsearch=doc_vectors)
        return response.query_response()