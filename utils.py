import os
import requests
from pathlib import Path
from azure.core.exceptions import HttpResponseError
from tenacity import retry, stop_after_attempt, wait_random_exponential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryCaptionResult,
    QueryAnswerResult,
    SemanticErrorMode,
    SemanticErrorReason,
    SemanticSearchResultsType,
    QueryType,
    VectorizedQuery,
    VectorQuery,
    VectorFilterMode,    
)
from azure.search.documents.indexes.models import (
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    HnswParameters,
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile
)
from dotenv import load_dotenv
load_dotenv("env")


def search_index(query, client, embedding_model):

    service_endpoint = os.environ["SEARCH_ENDPOINT"] 
    index_name = os.environ["SEARCH_INDEX_NAME"]
    key = os.environ["SEARCH_KEY"]
 
    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    vector_query = VectorizedQuery(vector=generate_embeddings(query, embedding_model, client), k_nearest_neighbors=3, fields="contentVector")

    results = list(search_client.search(  
        search_text=query,  
        vector_queries=[vector_query],
        select=["title", "content", "category"],
        query_type=QueryType.SEMANTIC, 
        semantic_configuration_name="default",
        query_caption=QueryCaptionType.EXTRACTIVE, 
        query_answer=QueryAnswerType.EXTRACTIVE,
        top=3
    ))

    return results

def search_index_semanticAndFallBack(query, client, embedding_model):

    service_endpoint = os.environ["SEARCH_ENDPOINT"] 
    index_name = os.environ["SEARCH_INDEX_NAME"]
    key = os.environ["SEARCH_KEY"]
 
    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    vector_query = VectorizedQuery(vector=generate_embeddings(query, embedding_model, client), k_nearest_neighbors=3, fields="contentVector")
    answer_context = []

    try:
        results = search_client.search(  
            search_text=query,  
            vector_queries=[vector_query],
            select=["title", "content", "category"],
            query_type=QueryType.SEMANTIC, 
            semantic_configuration_name="default",
            query_caption=QueryCaptionType.EXTRACTIVE, 
            query_answer=QueryAnswerType.EXTRACTIVE,
            top=3
        )

        for result in results:
            titles_and_content = {}
            titles_and_content["title"] = result["title"]
            titles_and_content["content"] = result["content"]
            answer_context.append(titles_and_content)
                    
    except HttpResponseError as e:
        if "Partial Content" in str(e):
            # Handle the 'Partial Content' case here
            print("Received partial content. Fall back to vector/text hybrid...\n")
            
            results = search_client.search(  
                search_text=query,  
                vector_queries=[vector_query],
                select=["title", "content", "category"],
                top=3
            )

            for result in results:
                titles_and_content = {}
                titles_and_content["title"] = result["title"]
                titles_and_content["content"] = result["content"]
                answer_context.append(titles_and_content)
                        
        else:
            # Handle other HTTP errors
            print(f"An HTTP error occurred: {e}")

    return answer_context


def gpt4v_query(messages, aoai_endpoint, aoai_api_key, api_version):

    headers = {"Content-Type": "application/json", "api-key": aoai_api_key}
    gpt4v_endpoint = f"{aoai_endpoint}/openai/deployments/gpt-4/extensions/chat/completions?api-version={api_version}" 

    payload = {
        "enhancements": {"ocr": {"enabled": True}, "grounding": {"enabled": True}},
        "messages": messages,
        "dataSources": [{
            "type": "AzureComputerVision",
            "parameters": {
                "endpoint": os.environ["VISION_ENDPOINT"],
                "key": os.environ["VISION_KEY"]
            }
        }],
        "temperature": 0,
        "top_p": 0.5,
        "max_tokens": 2000
    }

    try:
        response = requests.post(gpt4v_endpoint, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        print(response.content)  # print out the response content
        raise SystemExit(f"Failed to make the request. Error: {e}")

    return response


def gpt4_turbo_query(messages, client, chat_model):

    response = client.chat.completions.create(
    model=chat_model,
    temperature=0.0,
    response_format={ "type": "json_object" },
    messages=messages
    )
    return response


def append_conversation_history(messages, response, role):        
    new_conversation = {
        "role": role,
        "content": [
            {
                "type": "text",
                #"text": response.json()["choices"][0]["message"]["content"]
                "text": response.choices[0].message.content
            }
        ]
    }
    return messages.append(new_conversation)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def generate_embeddings(text, model, client):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


def list_blobs_download(blob_service_client, container_name, blob_name):
    container_client = blob_service_client.get_container_client(container=container_name)

    blob_list = container_client.list_blobs()

    for blob in blob_list:
        if blob.name.endswith((".jpeg", ".jpg", ".png")) and Path(blob.name).stem == Path(blob_name).stem:
            print(f"Download Image: {blob.name}")
            download_blob_to_file(blob_service_client, container_name, blob.name)


def list_blobs_titles_and_urls(blob_service_client, container_name, blob_name):
    container_client = blob_service_client.get_container_client(container=container_name)

    blob_list = container_client.list_blobs()
    image_titles_and_urls = {}

    for blob in blob_list:
        if blob.name.endswith((".jpeg", ".jpg", ".png")) and Path(blob.name).stem == Path(blob_name).stem:
            blob_client = blob_service_client.get_blob_client(container_name, blob.name)
            image_titles_and_urls["title"] = blob.name # Path(blob.name).stem 
            image_titles_and_urls["url"] = blob_client.url
            break

    return image_titles_and_urls


def list_blobs_titles_and_urls_ns(blob_service_client, container_name, blob_name, split_word):
    container_client = blob_service_client.get_container_client(container=container_name)

    blob_list = container_client.list_blobs()
    image_urls = []

    for blob in blob_list:
        image_titles_and_urls = {}
        if blob.name.endswith((".jpeg", ".jpg", ".png")) and blob.name.split(split_word)[0] == blob_name.split(split_word)[0]:
            blob_client = blob_service_client.get_blob_client(container_name, blob.name)
            image_titles_and_urls["title"] = blob.name # Path(blob.name).stem 
            image_titles_and_urls["url"] = blob_client.url
            image_urls.append(image_titles_and_urls)

    return image_urls


def list_blobs_urls(blob_service_client, container_name, blob_name, split_word):
    container_client = blob_service_client.get_container_client(container=container_name)

    blob_list = container_client.list_blobs()
    image_urls = []

    for blob in blob_list:
        if blob.name.endswith((".jpeg", ".jpg", ".png")) and Path(blob.name).stem.split(split_word)[0] == Path(blob_name).stem.split(split_word)[0]:
            blob_client = blob_service_client.get_blob_client(container_name, blob.name)
            image_url = blob_client.url
            image_urls.append(image_url)

    return image_urls


def download_blob_to_file(blob_service_client, container_name, blob_name):

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(blob_name, mode="wb") as sample_blob:
        download_stream = blob_client.download_blob()
        sample_blob.write(download_stream.readall())        
    return
