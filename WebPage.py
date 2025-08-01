import uuid 
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get("OPEN_AI_KEY")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# CONFIGURATION STREAMLIT 
st.set_page_config(
    page_title="Massinart's Assistant",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title(" 💬 CHATBOT FAQs ")
st.subheader("Bienvenue chez Massinart 👋")

logo_link="https://massinart.ma/cdn/shop/files/logo-Massinart-2025_1_245x_23048717-19c1-4cdd-bfda-c0e80272dcb0.svg?v=1736461547&width=300"
st.logo(
    logo_link,
    size="large",
    link="https://massinart.ma/"
)



#  RAG SESTEM SETUP 
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0.3,
    max_tokens=200
)

prompt = ChatPromptTemplate.from_messages([
    ('system', "Vous êtes un assistant de MASSINART qui répond aux questions des clients. "
     "Répondez en fonction du contexte et de l’historique de la conversation uniquement.\n"
     "#Instructions : Soyez précis et ne dépassez pas 3 phrases. "
     "- Si vous n’avez pas la réponse, répondez par : « Je n’ai pas une réponse précise.\n Contactez-nous directement : \n "
     "Service clients: 0522650148\n"
     "WhatsApp: 0707051494\n"
     "Email: contact@massinart.ma\n » "
     "-Si clint dit quelque chose de remaircement similaire vous devais répondrez par : « Je vous en prie . Vous pouvez nous contacter à tout moment .\n"
     "-SI LE CLIENT DEMANDE UN CONTACT , Voici les contacts de Massinart :\n"
     "Service clients: 0522650148\n"
     "WhatsApp: 0707051494\n"
     "Email: contact@massinart.ma\n" ),
    ('user', "Question: {input}\nContext: {context}")
])

chain = prompt | llm

loader = AsyncHtmlLoader("https://massinart.ma/pages/questions-frequentes-faq")
document = loader.load()
html2text = Html2TextTransformer()
documents2 = html2text.transform_documents(document)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["", " ", "\n", "\n\n"]
)
chunks = text_splitter.split_documents(documents2)

embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)

vectore_store = FAISS.from_documents(chunks, embedder)
retriever = vectore_store.as_retriever(search_kwargs={"k": 3})

# SESSION ID 
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
memory = st.session_state.memory

# INTERFACE UTILISATEUR 
query = st.chat_input("💬 Entrez votre question...")
with st.spinner("..."):
    if query:
        chat_history = memory.load_memory_variables({})['chat_history']
        last_history=memory.load_memory_variables({})['chat_history'][-6:]
        history = "\n\n".join([f"{msg.type.upper()}: {msg.content}" for msg in last_history])
        augmented_query=f"""
        {history} \n        
        {query} 
        """

        docs = retriever.invoke(augmented_query)
        context = "\n\n".join(doc.page_content for doc in docs)

        response = chain.invoke({
            'context': context,
            'input': augmented_query
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            st.markdown(response.content)

        memory.save_context(
            {"input": query},
            {"output": response.content}
        )

#  Historique visible
with st.expander("🗂 Historique complet"):
    if "memory" in st.session_state:
        chat_history = st.session_state.memory.load_memory_variables({})['chat_history']
        for i, m in enumerate(chat_history):
            role = m.type.capitalize()
            if role=="Human":      
                st.markdown(f"** Vous **: {m.content}")
            else:
                st.markdown(f"** Assistant **: {m.content}")

