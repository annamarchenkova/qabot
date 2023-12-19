# mypy: disable-error-code="truthy-bool, ignore-without-code"
import os
import yaml
import logging
import json
import requests

import openai
from openai.embeddings_utils import get_embeddings, get_embedding
from datetime import datetime
import hashlib
from tqdm.autonotebook import tqdm

# from pinecone.core.client.configuration import (
#     Configuration as OpenApiConfiguration,
# )

logging.basicConfig(level=logging.INFO)

from project_dirs import PROJECT_DIR  # DATA_DIR, OUTPUT_DIR


def load_config(cnf_dir=PROJECT_DIR, cnf_name="config.yml"):
    """
    load the yaml file
    """
    config_file = open(os.path.join(cnf_dir, cnf_name))
    return yaml.load(config_file, yaml.FullLoader)


def my_hash(s):
    # Return the MD5 hash of the input string as a hexadecimal string
    return hashlib.md5(s.encode()).hexdigest()


def prepare_for_pinecone(texts, engine):
    # print(engine)
    # Get the current UTC date and time
    now = datetime.utcnow()

    # Generate vector embeddings for each string in the input list, using the specified engine
    embeddings = get_embeddings(texts, engine=engine)

    # Create tuples of (hash, embedding, metadata) for each input string and its corresponding vector embedding
    # The my_hash() function is used to generate a unique hash for each string, and the datetime.utcnow() function is used to generate the current UTC date and time
    return [
        (
            my_hash(
                text
            ),  # A unique ID for each string, generated using the my_hash() function
            embedding,  # The vector embedding of the string
            dict(
                text=text, date_uploaded=now
            ),  # A dictionary of metadata, including the original text and the current UTC date and time
        )
        for text, embedding in zip(
            texts, embeddings
        )  # Iterate over each input string and its corresponding vector embedding
    ]


# Without namespaces that are not supported for the free tier
def upload_texts_to_pinecone(
    texts, engine, index, batch_size=None, show_progress_bar=False
):
    # Call the prepare_for_pinecone function to prepare the input texts for indexing
    total_upserted = 0
    if not batch_size:
        batch_size = len(texts)

    _range = range(0, len(texts), batch_size)
    for i in tqdm(_range) if show_progress_bar else _range:
        batch = texts[i : i + batch_size]
        prepared_texts = prepare_for_pinecone(batch, engine=engine)

        # print(prepared_texts)

        # Use the upsert() method of the index object to upload the prepared texts to Pinecone
        total_upserted += index.upsert(
            prepared_texts,
            # namespace=namespace
        )["upserted_count"]

    return total_upserted


def query_from_pinecone(query, engine, index, top_k=3):
    # get embedding from THE SAME embedder as the documents
    query_embedding = get_embedding(query, engine=engine)

    return index.query(
        vector=query_embedding,
        top_k=top_k,
        # namespace=namespace,
        include_metadata=True,  # gets the metadata (dates, text, etc)
    ).get("matches")


def delete_texts_from_pinecone(texts, index):
    # Compute the hash (id) for each text
    hashes = [hashlib.md5(text.encode()).hexdigest() for text in texts]

    # The ids parameter is used to specify the list of IDs (hashes) to delete
    return index.delete(ids=hashes)  # , namespace=namespace


def get_best_results_from_pinecone(query):
    payload = json.dumps(
        {
            "num_results": 2,
            "query": query,
            "re_runking_strategy": "none",
            # "namespace": namespace
        }
    )
    response = requests.post(
        "https://information-retrieval-hiaa.onrender.com/document/retreive",
        data=payload,
    )
    return response.json()["documents"][0]


def get_results_from_pinecone(query, engine, index, top_k=3, verbose=True):
    results_from_pinecone = query_from_pinecone(
        query, top_k=top_k, engine=engine, index=index
    )
    if not results_from_pinecone:
        return []

    if verbose:
        print("Query:", query)

    final_results = []

    if verbose:
        print("Document ID (Hash)\t\tRetrieval Score\tText")
    for result_from_pinecone in results_from_pinecone:
        final_results.append(result_from_pinecone)
        if verbose:
            print(
                f"{result_from_pinecone['id']}\t{result_from_pinecone['score']:.2f}\t{result_from_pinecone['metadata']['text'][:50]}"
            )

    return final_results


def test_prompt_openai(
    prompt, verbose=False, engine="gpt-note-crm", max_tokens=50, **kwargs
):
    # print(prompt)
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        # messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=max_tokens,
        **kwargs,
    )
    answer = response.choices[0].text
    if verbose:
        print(
            f"PROMPT:\n------\n{prompt}\n------\nRESPONSE\n------\n{answer}"
        )
    else:
        return answer


def get_texts_from_pinecone_results(results):
    texts = []
    for result in results:
        text = result["metadata"]["text"]
        texts.append(text)
    return texts


# use some chain of thought prompting to make it think through the reasoning.
# "Only answer this question and nothing else" is needed to not generate additional QA pairs.
def gen_Q_A(
    query,
    engine,
    index,
    specify_output=" ",
    qa_engine="azure",
    azure_engine="gpt-note-crm",
    n_results_to_use=2,
    max_tokens=100,
    verbose=False,
):
    results = get_results_from_pinecone(
        query,
        top_k=n_results_to_use,
        engine=engine,
        index=index,
        verbose=verbose,
    )
    texts = get_texts_from_pinecone_results(results)
    no_reply = "Sorry, I don't have this information."

    PROMPT = f"""
Only using the following context, answer the question. Only answer this question and nothing else. {specify_output}. If you can't unswer the question using the context say {no_reply}. 

Context: {" ".join(texts)}
Question: {query}
Answer:""".strip()

    if qa_engine == "azure":
        answer = test_prompt_openai(
            PROMPT,
            engine=azure_engine,
            verbose=verbose,
            max_tokens=max_tokens,
        )
    if qa_engine == "cohere":
        print("Sorry this functionality is not yet implemented")
        answer = " "

        # answer = test_prompt_openai(PROMPT, verbose=verbose, max_tokens=max_tokens)

    if no_reply.strip(".") in answer:
        answer = no_reply

    if "Question:" in answer:
        answer = answer[: answer.find("Question:")]

    return answer, texts
