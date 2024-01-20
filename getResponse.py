from langchain.vectorstores import Qdrant
import qdrant_client
from langchain.embeddings.cohere import CohereEmbeddings
import os
from dotenv import load_dotenv
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import Cohere
from IPython.display import display, Markdown, Latex


# Load the .env file
load_dotenv()

# Now you can get the API key from the environment
URL = os.getenv('URL')
API_KEY = os.getenv('QDRANT-API-KEY')

embeddings = CohereEmbeddings(model="embed-english-v3.0")

client = qdrant_client.QdrantClient(
    URL,
    api_key=API_KEY, # For Qdrant Cloud, None for local instance
)

doc_store = Qdrant(
    client=client, collection_name="pdf_summaries", 
    embeddings=embeddings,
)

retriever = doc_store.as_retriever(search_kwargs={"k": 30})

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

#create custom prompt for your use case
prompt_template="""As an expert in O Level Physics, your role is to provide clear, concise, and accurate responses to student inquiries. 
Use the course text retrieved to give precise explanations and ensure the information you share is reliable and well-founded. 
If a question falls outside the course content and your expertise, simply respond with "I don't know" — there's no need to speculate or offer uncertain answers. 
Your primary goal is to facilitate understanding by elucidating complex concepts, solving problems, and clarifying doubts while maintaining 
educational integrity. Respond in 50 words or less unless the question explicitly asks for longer answer.
----------------
{summaries}"""


messages = [
    SystemMessagePromptTemplate.from_template(prompt_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

chain_type_kwargs = {"prompt": prompt}

llm=Cohere(model="command-nightly")


chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)
