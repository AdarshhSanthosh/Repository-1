# Import necessary libraries
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as VectorStorePinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.tools import DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
import gradio as gr
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY="pcsk_5ghur7_3N246zk7wLTQKGC2vHFsPBZNSCsApzAbqJujv3BwSN2ucueVpfr3VP3wecuCecB"
PINECONE_ENV = os.getenv("PINECONE_ENV")  # Add environment variable for Pinecone environment

# Configuration constants
DATA_PATH = r"data"

# Step 1: Initialize embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "example-index2"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENV
        )
    )

index = pc.Index(index_name)

# Step 2: Load and process PDF documents
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

# Step 3: Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len
)
documents = text_splitter.split_documents(raw_documents)

# Step 4: Embed document chunks and insert into Pinecone
for i, chunk in enumerate(documents):
    chunk_embedding = embeddings_model.embed_documents([chunk.page_content])
    metadata = {"text": chunk.page_content}  # Add metadata with the document text
    index.upsert(vectors=[(f"doc-{i}", chunk_embedding[0], metadata)])

def retrieve_relevant_chunks(query):
    # Generate query embedding
    query_embedding = embeddings_model.embed_query(query)
    
    # Retrieve top-k results from Pinecone using keyword arguments
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    
    # Extract metadata for relevant documents
    return results["matches"]

# Step 6: Initialize the ChatOpenAI model
llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")

# Step 6: Set up the DuckDuckGo search tool
search = DuckDuckGoSearchResults()
search_tool = Tool(
    name="duckduck",
    description="A web search engine. Use this to as a search engine for general queries.",
    func=search.run,
)

# Prepare tools for the agent
tools = [search_tool]

# Create ReAct agent for general queries
react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate(
    template=react_template,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

# Step 7: Define response streaming function
def stream_response(message, history):
    try:
        # Attempt to retrieve relevant chunks from Pinecone
        matches = retrieve_relevant_chunks(message)

        # Combine retrieved chunks into a knowledge base
        knowledge = ""
        for match in matches:
            if "metadata" in match and "text" in match["metadata"]:
                knowledge += match["metadata"]["text"] + "\n\n"

        if not knowledge.strip():  # If no knowledge is found, use the search tool
            response = agent_executor.invoke({"input": message})
            yield response.get("output", "No response from search tool.")
            return

        # Construct the RAG prompt if knowledge is found
        rag_prompt = f"""
        You are a highly knowledgeable and helpful assistant.
        Your role is to answer questions and assist with tasks to the best of your ability,
        using your internal knowledge and reasoning skills.
        Provide clear, accurate, and concise responses tailored to the user's queries.
        Feel free to ask clarifying questions if needed to provide the best assistance.

        Knowledge base: {knowledge}

        The question: {message}
        """

        # Stream the response to the Gradio app
        partial_message = ""
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message
    except Exception as e:
        yield f"An error occurred: {str(e)}"

# Step 8: Create the Gradio chatbot interface
chatbot = gr.ChatInterface(
    fn=stream_response,
    textbox=gr.Textbox(
        placeholder="Send to the LLM... (The model decides between search or knowledge base)",
        container=False,
        autoscroll=True,
        scale=7
    ),
)

# Step 9: Launch the chatbot
chatbot.launch(share=True)  # Set share=True for public link
