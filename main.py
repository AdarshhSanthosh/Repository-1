# Import necessary libraries
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as VectorStorePinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import gradio as gr
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PINECONE_ENV = os.getenv("PINECONE_ENV")  # Add environment variable for Pinecone environment

# Configuration constants
DATA_PATH = r"data"

# Step 1: Initialize embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "example-index3"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=us-east-1
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

def is_realtime_query(query):
    # Define patterns for detecting real-time or time-sensitive queries
    time_sensitive_keywords = ["today", "latest", "current", "now", "2024", "2025"]
    time_sensitive_patterns = r"|".join(time_sensitive_keywords)
    
    # Check if query matches any of the patterns
    return bool(re.search(time_sensitive_patterns, query, re.IGNORECASE))

# Step 5: Initialize the ChatOpenAI model
llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")

# Step 6: Set up the DuckDuckGo search tool
search = DuckDuckGoSearchResults()
search_tool = Tool(
    name="duckduck",
    description="A web search engine. Use this to as a search engine for general queries.",
    func=search.run,
)

# Step 7: Set up the Pinecone DB tool
def pinecone_db_tool(query):
    matches = retrieve_relevant_chunks(query)
    knowledge = ""
    for match in matches:
        if "metadata" in match and "text" in match["metadata"]:
            knowledge += match["metadata"]["text"] + "\n\n"
    return knowledge

pinecone_tool = Tool(
    name="pinecone_db",
    description="A tool to retrieve relevant information from local documents stored in Pinecone.",
    func=pinecone_db_tool,
)

# Prepare tools for the agent
tools = [pinecone_tool, search_tool]  # Prioritize Pinecone tool over search tool

# Step 8: Set up memory for the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Step 9: Create ReAct agent for general queries
react_template = """Answer the following questions as best you can. You have access to the following tools when i ask a question, first you should check it on the pinecone database.you should also keep the history of chats. and when i ask like "what was my n'th question you should reply." :

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
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory
)

# Step 10: Define response streaming function
def stream_response(message, history):
    try:
        # Load the chat history from memory
        chat_history = memory.load_memory_variables({})["chat_history"]

        # Check if the user is asking about their name
        if "what is my name" in message.lower():
            # Retrieve the user's name from memory
            for msg in chat_history:
                if "my name is" in msg.content.lower():
                    name = msg.content.split("my name is")[1].strip()
                    yield f"Your name is {name}."
                    return
            yield "I don't know your name yet. Please tell me your name."
            return

        # Check if the user is asking about their first question
        if "what was my first question" in message.lower():
            if len(chat_history) > 0:
                first_question = chat_history[0].content
                yield f"Your first question was: {first_question}"
            else:
                yield "You haven't asked any questions yet."
            return

        # Use the agent executor to handle the query
        response = agent_executor.invoke({"input": message})
        
        # Update the memory with the new interaction
        memory.save_context({"input": message}, {"output": response.get("output", "No response from the agent.")})
        
        # Yield the response
        yield response.get("output", "No response from the agent.")

    except Exception as e:
        yield f"An error occurred: {str(e)}"
# Step 11: Create the Gradio chatbot interface
chatbot = gr.ChatInterface(
    fn=stream_response,
    textbox=gr.Textbox(
        placeholder="Send to the LLM... (The model decides between search or knowledge base)",
        container=False,
        autoscroll=True,
        scale=7
    ),
)

# Step 12: Launch the chatbot
chatbot.launch(share=True)  # Set share=True for public link