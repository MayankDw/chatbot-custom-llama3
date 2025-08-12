import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain import hub

# https://ollama.com/blog/embedding-models
# https://github.com/ollama/ollama

# ollama serve
# sudo service ollama stop
# ollama pull llama3
# ollama pull moondream
# /bin/python3 -m streamlit run chat.py

# Modelo a usar
MODEL = "llama3" 
#MODEL = "moondream"
#MODEL = "phi3:medium"

@st.cache_data
def load_data(url):
    # load URL
    loader = WebBaseLoader(url)
    data = loader.load()

    # Divide the text in parts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    return all_splits

def extract_response(response):
    # get results in dictionary 
    if isinstance(response, dict) and 'result' in response:
        return response['result']
    return response

def main():
    """
    Main function for the chat application with Llama3.

    This function initializes the conversation history, takes user input for the URL and custom question,
    and performs the chatbot interaction with Llama3. It also processes and displays the conversation history.

    Returns:
        None
    """
    st.title("Chat with llama")
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []

    
    url = st.text_input("Enter the URL to scrape", "https://d2l.ai/chapter_linear-regression/generalization.html#underfitting-or-overfitting")

    
    custom_question = st.text_input("You: ", "What is the main idea about the text?")

    
    if st.button("Send"):
        with st.spinner("Fetching data..."):
            
            all_splits = load_data(url)

            
            vectorstore = Chroma.from_documents(
                documents=all_splits, embedding=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf", gpt4all_kwargs={'allow_download': True})
            )

            
            llm = Ollama(
                model=MODEL,
                verbose=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )

            
            QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

            
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            )

            
            question = custom_question.strip() if custom_question else f"What is the main idea of {url}?"

            
            response = qa_chain.invoke({"query": question})
            
            def extract_response(response):
                
                if isinstance(response, dict) and 'result' in response:
                    cleaned_response = response['result']
                else:
                    cleaned_response = response

              m 
                cleaned_response = cleaned_response.replace("[/INST]<<SYS>>", "")
                cleaned_response = cleaned_response.replace("[SYS]>", "")
                cleaned_response = cleaned_response.replace("[SYS]", "")
                cleaned_response = cleaned_response.replace("[/INST]", "")
                cleaned_response = cleaned_response.replace("[/STUDENT]", "")
                cleaned_response = cleaned_response.replace("[SYS>>", "")
                cleaned_response = cleaned_response.replace("<<]", "")  
                cleaned_response = cleaned_response.replace("[INST]<<SYS>>", "")
                cleaned_response = cleaned_response.replace("<<</SYS>>", "")
                cleaned_response = cleaned_response.replace("[/SYS>>]", "")
                cleaned_response = cleaned_response.replace("<<SYS]]", "")
                cleaned_response = cleaned_response.replace("[HUMAN]<<SYS>> ", "")
                cleaned_response = cleaned_response.replace("[/HUMAN]", "")
                cleaned_response = cleaned_response.replace("[/SYS]>", "")
                
                return cleaned_response.strip()


          
            cleaned_response = extract_response(response)

         
            st.session_state['conversation_history'].append(("You:", custom_question))
            st.session_state['conversation_history'].append(("Llama3:", cleaned_response))

     
    for sender, message in st.session_state['conversation_history']:
        if sender == "You:":
            st.text_input(sender, message, key=message)
        else:
            
            st.text_area(sender, message, height=100, key=message)

if __name__ == "__main__":
    main()
