from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from streamlit_chat import message as st_message


def main():
    user_question = None
    text = ""

    load_dotenv()
    
    st.set_page_config(layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	                      initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed",
                        page_title="Ask your Document💬", 
                        page_icon=":fire:")
    st.header("Ask your Document 💬")
    hide_menu = """
    <style>footer {visibility: hidden;}
    footer:before {content:'Made with ❤️ by Waqas. Work in Progress'; visibility: visible;
      display: block;
      position: relative;
      #background-color: red;
      padding: 5px;
      top: 2px;
    }
    </style>
    """
    st.markdown(hide_menu, unsafe_allow_html=True) 

    if "history" not in st.session_state:
      st.session_state.history = []

    # upload file
    pdfs = st.file_uploader("Upload upto 4 PDF's 📄📄📄", type="pdf", accept_multiple_files=True)
    for pdf in pdfs:
    
      # extract the text
      if pdf is not None:
        print(pdf.name)
        try:
          pdf_reader = PdfReader(pdf)
          for page in pdf_reader.pages:
            text += page.extract_text()
          print("text", text[0:100])
        except:
          print('An exception occurred')
        # split into chunks
        text_splitter = CharacterTextSplitter(
          separator="\n",
          chunk_size=1000,
          chunk_overlap=250,
          length_function=len
        )

        if text:
          chunks = text_splitter.split_text(text)
          # create embeddings
          embeddings = OpenAIEmbeddings()
          knowledge_base = FAISS.from_texts(chunks, embeddings)
      else:
        st.error(f"Sorry the file {pdf.name} cannot be processed. Try another document.")

    if pdfs and knowledge_base is not None:
      # show user input
      if user_question is None:
        chat_history = []

        user_question = st.text_input("Ask a question in any language about your Document:")
      # resbox = st.empty()

        if user_question:          
          llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()],model_name="gpt-3.5-turbo", temperature=0.1)
          # chain = load_qa_chain(llm, chain_type="stuff")
          qa = ConversationalRetrievalChain.from_llm(llm, knowledge_base.as_retriever())
          with get_openai_callback() as cb:
            response = qa({"question": user_question, "chat_history": chat_history})
            chat_history.append((user_question, response['answer']))
            print(cb)
            st.success(response["answer"]) 
        
          st.session_state.history.append({"message": user_question, "is_user": True})
          st.session_state.history.append({"message": response['answer'], "is_user": False})
          with st.expander("Chat History"):  
            for i, chat in enumerate(st.session_state.history):
              st_message(**chat, key=str(i))

if __name__ == '__main__':
    main()