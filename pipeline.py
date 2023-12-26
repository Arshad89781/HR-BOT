from data_extraction import PDFTextExtractor,PineconeVectorDatabase

class ExtractionAndVectorStoringTask:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def vector_storing(self):
        text_extractor = PDFTextExtractor(folder_path=self.folder_path)
        text_data = text_extractor.extract_text()

        VectorData = PineconeVectorDatabase(text_chunks=text_data)
        VectorData.store_vector()