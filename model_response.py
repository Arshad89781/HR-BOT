from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from config import PINECONE_API_ENV,PINECONE_API_KEY,INDEX_NAME,LLM_MODEL_PATH
from prompt import prompt_template



class LLMResponse:
    def __init__(self, query,docsearch):
        self.query = query
        self.docsearch = docsearch

    def query_response(self):
        PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs={"prompt": PROMPT}

        llm=CTransformers(model=LLM_MODEL_PATH,
                        model_type="llama",
                        config={'max_new_tokens':512,
                                'temperature':0})
        qa=RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=self.docsearch.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True, 
            chain_type_kwargs=chain_type_kwargs)
        result=qa({"query": self.query})
        print(result["result"])
        return result["result"]