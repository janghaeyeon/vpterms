from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st

#Loader
loader = PyPDFLoader("ya.pdf")
pages = loader.load_and_split()

#Split
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 300,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
)
texts = text_splitter.split_documents(pages)

#Embedding
embeddings_model = OpenAIEmbeddings()

# load it into Chroma
db = Chroma.from_documents(texts, embeddings_model)

#Question
st.title('브이피 약관 문의 LLM')

question = st.text_input('브이피 약관에 대해 문의할 내용을 입력하세요.')
#question = "이용자가 회사에 대하여 분쟁처리를 신청한 경우 몇일이내 처리해야하나?"
if st.button('문의하기'):
    with st.spinner('생각중.....'):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
        result = qa_chain({"query": question})
        #print(result)
        st.write(result)

# streamlit run 5-7.py
 
 


        
       