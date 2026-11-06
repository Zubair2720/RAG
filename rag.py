import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page Setup

st.set_page_config(page_title='PDF Rag Agent',page_icon='💻',layout='wide')
st.title('📄 PDF Rag Agent')
st.caption('Upload a PDF,then ask Question - Answer come only from document')

# Sidebar

with st.sidebar:
    st.header('Setting')
    API_Key=st.text_input('Gemini API Key',type='password',value=os.getenv("Gemini_API_Key",""))
    model_select=st.selectbox('Chat Model',['gemini-3.5-flash-lite','gemini-2.5-flash'])
    chunk_size=st.slider("Chunk Size",500,2000,1000,step=100)
    chunk_overlap=st.slider('Chunk Overlap',0,400,150,step=50)
    top_k=st.slider('Retrieved Chunks(k)',2,10,4)

if not API_Key:
    st.info('Please Enter API Key in the sidebar')
    st.stop()

os.environ['GOOGLE_API_KEY'] = API_Key
# File Upload + indexing

file_upload=st.file_uploader('Upload a file',type=['pdf'])

def build_vectorstore(
    pdf_bytes: bytes,
    size: int,
    overlap: int,
    api_key: str
) -> FAISS:
    """Load PDF,split into chunk,embed,and store in Faiss"""
    with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as tmp:
        tmp.write(pdf_bytes)
        tmp_path=tmp.name
    try:
        docs=PyPDFLoader(tmp_path).load()
    finally:
        os.unlink(tmp_path)

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=['\n\n','\n','.',' ','']
    )
    chunks=splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,   # function ka parameter
    )
    return FAISS.from_documents(chunks,embeddings)

if file_upload is not None:
    file_sig=(file_upload.name,
    file_upload.size,
    chunk_size,
    chunk_overlap)
    if st.session_state.get("file_sig") != file_sig:
        with st.spinner('Reading and Indexing PDF'):
            st.session_state.vectorstore = build_vectorstore(
    file_upload.getvalue(),
    chunk_size,
    chunk_overlap,
    API_Key
)
            
            st.session_state.file_sig=file_sig
            st.session_state.messages=[]
        st.success(f"Indexed **{file_upload}** and ask away")

# RAG

RAG_Prompt=ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI Assistant that answer questions strictly using the"
     "Provided context from a PDF document.\n"
     "Rules:\n"
     "1. Answer only from the context below.\n"
     "2. If the answer is not in context, say: "
     "\" I could not find in that document.\"\n"
     "3. Cite the page number(s) you used, e.g. (p. 3).\n\n"
     "context:\n{context}"),
     ("human","{question}")   
])

def format_docs(docs) -> str:
    """Join retrieved chunks, tagging each with its page number."""

    return "\n\n".join(
        f"[Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    )

def get_chain(vectorstore:FAISS,model:str,k:int):
    retriever=vectorstore.as_retriever(search_kwargs={"k":k})
    llm = ChatGoogleGenerativeAI(model=model,temperature=0,google_api_key=API_Key)
    return(
        {"context":retriever|format_docs,"question":RunnablePassthrough()}
        |RAG_Prompt
        |llm
        |StrOutputParser()
    ),retriever

# Chat Interface

st.divider()
st.subheader("💬 Chat With Your PDF")

if file_upload is None:
    st.info("👆 Please upload a PDF first.")

elif "vectorstore" not in st.session_state:
    st.warning("Please wait! PDF is being indexed.")

else:
    # Get RAG Chain
    chain, retriever = get_chain(
        st.session_state.vectorstore,
        model_select,
        top_k
    )

    # Display Chat History
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    user_question = st.chat_input(
        "Ask anything about your PDF..."
    )

    if user_question:

        # Display User Question
        with st.chat_message("user"):
            st.markdown(user_question)

        # Save User Question
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        # Generate AI Answer
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = chain.invoke(user_question)

                st.markdown(response)

        # Save AI Response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })


