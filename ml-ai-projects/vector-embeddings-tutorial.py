# --- Configuration Variables ---
ASTRA_DB_SECURE_BUNDLE_PATH = "bundle_path"  # Path to Astra DB SCB zip file
ASTRA_DB_APPLICATION_TOKEN = "token_value"   # Astra DB application token
ASTRA_DB_CLIENT_ID = "client_id_value"       # Astra DB client ID
ASTRA_DB_CLIENT_SECRET = "token_secret"      # Astra DB client secret
ASTRA_DB_KEYSPACE = "keyspace_name"          # Keyspace name in Astra DB
OPENAI_API_KEY = "openai_api_key"            # OpenAI API key for embeddings

# --- Import Required Libraries ---
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.vectorstores.cassandra import Cassandra
from langchain.llms import OpenAI

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset

# --- Initialize LLM and Embedding Function ---
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)  # LLM for answering questions
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)  # Embedding function for text

# --- Set Up Cassandra (Astra DB) Connection ---
# Create authentication provider using Astra DB credentials
auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_SECRET)
# Define cloud configuration for secure connection
cloud = {'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH}
# Initialize Cassandra cluster with secure bundle and authentication
cluster = Cluster(cloud=cloud, auth_provider=auth_provider)
# Establish session with Astra DB
astraSession = cluster.connect()

# --- Initialize Cassandra Vector Store ---
my_cassandra_vstore = Cassandra(
    session=astraSession,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name="table_name",  # Table to store vectors
    embedding_function=embeddings
)

"""
# --- (Optional) Load and Insert Data into Vector Store ---
# Loads a subset of the Wikipedia dataset, extracts first 50 article titles,
# generates embeddings, and inserts them into the Cassandra vector store.

# print("Loading huggingface dataset...")
# dataset = load_dataset("wikipedia", "20220301.en", split="train[:1000]")
# headlines = dataset["title"][:50]
# print("\nGenerating embeddings...")
# my_cassandra_vstore.add_texts(headlines)
# print("Inserted %i headlines.\n" % len(headlines))
"""

# --- Set Up Vector Index for Querying ---
vector_index = my_cassandra_vstore

# --- Interactive Query Loop ---
first_question = True
while True:
    # Prompt user for input
    prompt = "\nEnter question or type 'quit' to exit: " if first_question else "\nEnter next question or type 'quit' to exit: "
    query_text = input(prompt)
    if query_text.lower() == 'quit':
        break
    first_question = False

    print("QUESTION: \"%s\"\n" % query_text)
    try:
        # Attempt to query with LLM (for generative answers)
        result = vector_index.query(query_text, llm=llm)
    except TypeError as e:
        # Fallback: Retry without LLM if not supported
        if "unexpected keyword argument" in str(e):
            result = vector_index.query(query_text)
        else:
            raise
    # Format and display the answer
    answer = result.strip() if isinstance(result, str) else str(result).strip()
    print("ANSWER: \"%s\"\n" % answer)

    # Retrieve and display top relevant documents with similarity scores
    print("DOCUMENTS BY RELEVANCE: ")
    for doc, score in my_cassandra_vstore.similarity_search_with_score(query_text, k=4):
        print("  %0.4f \"%s\"" % (score, doc.page_content))
