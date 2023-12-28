from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from config import PINECONE_API_ENV,PINECONE_API_KEY,INDEX_NAME,LLM_MODEL_PATH
from prompt import prompt_template
from langchain_google_genai import ChatGoogleGenerativeAI


class LLMResponse:
    def __init__(self, query,docsearch):
        self.query = query
        self.docsearch = docsearch
    
    def get_selected_llm(self,llm_name = "gemini pro"):
        if llm_name == "gemini pro":
            llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                    temperature=0.3,google_api_key="AIzaSyDckAhloHvKIeoeqA4HeB9fHUPrsRp4WiI")
            return llm
        elif llm_name == "llama":
            llm=CTransformers(model=LLM_MODEL_PATH,
                        model_type="llama",
                        config={'max_new_tokens':512,
                                'temperature':0})
            return llm

    def query_response(self):
        PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs={"prompt": PROMPT}
        
        llm = self.get_selected_llm(llm_name="gemini pro")

        qa=RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=self.docsearch.as_retriever(),
            return_source_documents=True, 
            chain_type_kwargs=chain_type_kwargs)
        result=qa({"query": self.query})
        print(result["result"])
        return result["result"]