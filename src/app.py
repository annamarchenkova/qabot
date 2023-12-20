import streamlit as st
import os
import openai
from langchain.text_splitter import CharacterTextSplitter
import pinecone

from utils import (
    load_config,
    upload_texts_to_pinecone,
    gen_Q_A
)
from project_dirs import PROJECT_DIR, DATA_DIR

def main():

    st.title("Answering questios about your documents")

    # Description of the app
    st.markdown("""This app allows you to load your documents and then answer 
                questions using the information in these documents.""")
        
    # Input field for documents' folder
    folder_name = st.text_area(
        "Enter the folder name where your documents are stored",
        height=20,
        )
    # # Input field 
    # pinecone_key = st.text_area(
    #     "Enter your Pinecone key",
    #     height=20,
    #     )
    # # Input field 
    # openai.api_key = st.text_area(
    #     "Enter your Azure OpenAI key",
    #     height=20,
    #     )

    cnf = load_config(cnf_dir=PROJECT_DIR, cnf_name="config.yml")

    pinecone_key = open(os.path.join(PROJECT_DIR, "keys", "pinecone_key.txt"), "r").read().strip("\n")
    pinecone_key = open(os.path.join(PROJECT_DIR, "keys", cnf["pinecone_key_file"]), "r").read().strip("\n")
    openai_api_key = open(os.path.join(PROJECT_DIR, "keys", cnf["azure_openai_key_file"]), "r").read().strip("\n")

    deployment_name = cnf['deployment_name']
    
    # PINECONE index
    index_name = cnf['index_name']
    pinecone_engine = cnf['engine']

    t_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=500, 
        chunk_overlap=10,
        )

    # Button to trigger document upload
    if st.button("Upload text documents"):
        folder_path = os.path.join(DATA_DIR, folder_name)
        for filename in os.listdir(folder_path):
            if (
                (filename.endswith('.txt')) | 
                (filename.endswith('.docx'))
                ):
                filepath = os.path.join(DATA_DIR, folder_name, filename)
                with open(filepath, encoding='utf-8') as f:
                    text = f.read()
                
                texts = t_splitter.split_text(text)
                texts = [i for i in texts if len(i) > cnf['min_chunk_len']]
                upload_texts_to_pinecone(
                    texts, 
                    engine=pinecone_engine, 
                    pinecone_key=pinecone_key,
                    index_name=index_name,
                    batch_size=5, 
                    show_progress_bar=True,
                    )
                st.success('Uploaded document {filename}')

        st.success('All documents uploaded successfully!')

    # Question
    question = st.text_area(
        "Enter your question",
        height=100,
        )
    specify_output = st.text_area(
        "Specify desired output. Ex: Give a one-sentence answer.",
        height=100,
        )
    # n_results_to_use
    n_results_to_use = st.text_area(
        "Number of chunk extractions used for the answer; from 1 to 10",
        height=20,
        )
    try:
        n_results_to_use = int(n_results_to_use)
        if not 1<= n_results_to_use <= 10:
            st.error('Enter a number from 1 to 10')
    except:
        st.error('Enter a number from 1 to 10')

    # max_tokens
    max_tokens = st.text_area(
        "Enter maximum number of tokens in the responce; from 1 to 200",
        height=20,
        )
    try:
        max_tokens = int(max_tokens)
        if not 1<= max_tokens <= 200:
            st.error('Enter a number from 1 to 200')
    except:
        st.error('Enter a number from 1 to 200')

    # Answer
    if st.button("Get answer"):
        answer, chunks = gen_Q_A(
                    query=question,
                    engine=pinecone_engine,
                    index_name=index_name,
                    pinecone_key=pinecone_key,
                    openai_api_key=openai_api_key,
                    specify_output=specify_output,
                    qa_engine=openai.api_type,
                    azure_engine=deployment_name,
                    n_results_to_use=n_results_to_use,
                    max_tokens=max_tokens,
                    verbose=False,
                    )
        
        st.markdown(f"**Answer**: {answer}")
        st.markdown(f"**Text chunks**: {chunks}")
   

if __name__ == "__main__":
    main()
