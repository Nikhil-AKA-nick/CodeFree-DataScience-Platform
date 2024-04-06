import streamlit as st
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_openai.chat_models import ChatOpenAI
import openai
import dotenv
# from dotenv import load_dotenv


import os


os.environ['OPENAI_API_KEY'] = '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'

# load_dotenv()

os.environ["GOOGLE_API_KEY"] = "AIzaSyBMzHCmzXoGoO0kUFiUfYZxIAgb7JKifok"


st.set_page_config("SQL Agent", page_icon = "📑")

st.header('Chat with SQL Database')


llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

open_ai_llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0.1)

if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content = "Hello, I am your aweomse and cool SQL bot who has expertise in Data Analysis using SQL.")
            ]


options = st.sidebar.selectbox("Select one of the options below to upload the Database", options = ["Local", "PostgreSQL", "MySQL"])

if options == "Local":

    input_from_user = st.sidebar.file_uploader("Upload your Database file here", type=["db"])
    
    ##Getting the Database
    if input_from_user:
        if not os.path.exists(input_from_user.name):
            # Save the uploaded file in the current working directory
            with open(os.path.join(os.getcwd(), input_from_user.name), "wb") as f:
                f.write(input_from_user.getbuffer())
                
        db = SQLDatabase.from_uri(f"sqlite:///{input_from_user.name}")
        


        user_query = st.chat_input("Enter your Question here...")
            
            
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
                
        agent = create_sql_agent(llm=llm, toolkit=toolkit, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
                #openai_agent = create_sql_agent(llm=open_ai_llm, toolkit=toolkit, agent_type=AgentType.OPENAI_FUNCTIONS, verbose=True)
            
        if user_query:
                
                gemini_response = agent.run(user_query)
                #openai_response_with_functional_calling = openai_agent.run(user_query)
                
                ##Appending the user_input and response to the chat_history
                st.session_state.chat_history.append(HumanMessage(content = user_query))
                st.session_state.chat_history.append(AIMessage(content = gemini_response))
                
                
                ##Displaying the chat history in the Application:
                for message in st.session_state.chat_history:
                    if isinstance(message, AIMessage):
                    
                        with st.chat_message("AI"):
                            st.write(message.content)
                        
                    elif isinstance(message, HumanMessage):
                        
                        with st.chat_message("Human"):
                            st.write(message.content)

elif options == "PostgreSQL":
    username = st.sidebar.text_input("Enter the username")
    password = st.sidebar.text_input("Enter the Password")
    host = st.sidebar.text_input("Enter the host")
    port = st.sidebar.text_input("Enter the port")
    database = st.sidebar.text_input("Enter the Datbase name")
    
    db_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    
    db = SQLDatabase.from_uri(db_uri)
    
    user_query = st.chat_input("Enter your Question here...")
            
            
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
                
    #agent = create_sql_agent(llm=llm, toolkit=toolkit, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    openai_agent = create_sql_agent(llm=open_ai_llm, toolkit=toolkit, agent_type=AgentType.OPENAI_FUNCTIONS, verbose=True)
            
    if user_query:
                
                #gemini_response = agent.run(user_query)
                openai_response_with_functional_calling = openai_agent.run(user_query)
                
                ##Appending the user_input and response to the chat_history
                st.session_state.chat_history.append(HumanMessage(content = user_query))
                st.session_state.chat_history.append(AIMessage(content = openai_response_with_functional_calling))
                
                
                ##Displaying the chat history in the Application:
                for message in st.session_state.chat_history:
                    if isinstance(message, AIMessage):
                    
                        with st.chat_message("AI"):
                            st.write(message.content)
                        
                    elif isinstance(message, HumanMessage):
                        
                        with st.chat_message("Human"):
                            st.write(message.content)
    
elif options == "MySQL":
    db_user = st.text_input("Enter the User")
    db_password = st.text_input("Enter the Password")
    db_host = st.text_input("Enter the host")
    db_name = st.text_input("Enter the Database Name")
    
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
    
    user_query = st.chat_input("Enter your Question here...")
            
            
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
                
    agent = create_sql_agent(llm=llm, toolkit=toolkit, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
                #openai_agent = create_sql_agent(llm=open_ai_llm, toolkit=toolkit, agent_type=AgentType.OPENAI_FUNCTIONS, verbose=True)
            
    if user_query:
                
                gemini_response = agent.run(user_query)
                #openai_response_with_functional_calling = openai_agent.run(user_query)
                
                ##Appending the user_input and response to the chat_history
                st.session_state.chat_history.append(HumanMessage(content = user_query))
                st.session_state.chat_history.append(AIMessage(content = gemini_response))
                
                
                ##Displaying the chat history in the Application:
                for message in st.session_state.chat_history:
                    if isinstance(message, AIMessage):
                    
                        with st.chat_message("AI"):
                            st.write(message.content)
                        
                    elif isinstance(message, HumanMessage):
                        
                        with st.chat_message("Human"):
                            st.write(message.content)