from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from streamlit_chat import message as st_message


def main():
    load_dotenv()
    
    st.set_page_config(layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	                      initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed",
                        page_title="Ask your Document 💬", 
                        page_icon="💬")
    st.header("Ask your Document 💬")
    

    hide_streamlit_style=""" <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    if "history" not in st.session_state:
      st.session_state.history = []

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:

      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
        
        st.session_state.history.append({"message": user_question, "is_user": True})
        st.session_state.history.append({"message": response, "is_user": False})

        for i, chat in enumerate(st.session_state.history):
          st_message(**chat, key=str(i))

if __name__ == '__main__':
    main()