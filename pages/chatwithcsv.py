import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI

import streamlit as st
import openai
import tempfile
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
# import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment as am
import streamlit_authenticator as stauth
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI


import os

os.environ['OPENAI_API_KEY'] = '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'


# st.title("Chat With CSV")


selected = option_menu(None, ["Analyze", "Vizualize", "Data Scrubbing"], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
# selected



def handle_uploaded_file(uploaded_file):
    """
    Handles the uploaded CSV file and displays it with summary statistics.

    Args:
        uploaded_file (UploadedFile): The uploaded CSV file object.
    """
    try:
        # Display a placeholder message before upload
        placeholder = st.empty()
        placeholder.text("Drag and drop your CSV file here, or click to browse.")

        # Progress bar for upload
        progress = st.progress(0)

        if uploaded_file is not None:
            progress.progress(20)  # Update progress on file selection

            # Read the CSV data with potential encoding and delimiter options
            df = pd.read_csv(
                uploaded_file,
                encoding="utf-8",  # Adjust encoding as needed
                delimiter=",",  # Adjust delimiter as needed
            )

            progress.progress(100)  # Update progress on completion

            # Display success message
            st.success("File uploaded successfully!")

            # Remove placeholder text
            placeholder.empty()

            # Display the DataFrame
            st.dataframe(df)

            # Display summary statistics
            st.write(df.describe(include="all"))
            
            return df

        else:
            progress.empty()  # Remove progress bar if no file selected

    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")


def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.read())
    return temp_file.name

def read_csv_with_date_format(file_path):
    return pd.read_csv(file_path, infer_datetime_format=True)


def load_default_csv(selected_csv):
    default_csv_paths = {
        "Sample Dataset": "pages\sample.csv"
        # Add more default CSVs with user-friendly names and paths
    }
    return pd.read_csv(default_csv_paths[selected_csv])

def execute_generated_code(code, df):
    try:
        # Execute the code to generate the plots
        exec(code, globals(), {'df': df})

        # Display each plot using st.pyplot()
        st.subheader("Generated Plots:")
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Find all Matplotlib figures created by the code
        generated_plots = [plt.figure(i) for i in plt.get_fignums()]

        # Display each plot separately
        for idx, plot in enumerate(generated_plots):
            st.subheader(f"Plot {idx + 1}:")
            st.pyplot(plot)

            # Close the Matplotlib plot
            plt.close(plot)

    except Exception as e:
        st.error(f"Error executing code: {e}")

if selected=="Analyze":
    st.title("Analyze CSV")

    def main():
        """
        The main function of the Streamlit app.
        """
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            with st.spinner("Loading data..."):  # Display spinner while processing
                df = handle_uploaded_file(uploaded_file)
                
                
                
        else:
            st.info("Please upload a CSV file to start.")
            df = pd.DataFrame()
        
            
        
        
        agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),df,verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,)

        
        user_input = st.text_input("Enter your text:", "Type here...")
        submit=st.button("Ask the question")
        
        
        if submit and input:
            st.write(agent.invoke(user_input))
            
        st.write(agent.run(user_input))

    if __name__ == "__main__":
        main()


if selected=="Vizualize":
    def main():
        st.title("Vizualize the CSV")
        csv_file = st.file_uploader("Choose a CSV file", type="csv")
        
        selected_default_csv = st.radio("Select Default CSV:", options=["Sample Dataset"])
        model_name = st.radio("Choose a Model...", options=["gpt-3.5-turbo-1106", "gpt-4", "gpt-4-32k", "gpt-4-vision-preview"])

        
        if csv_file is None:
            existing_csv_path = "sample.csv"
            st.subheader("Loaded Data:")
            df = load_default_csv(selected_default_csv)
            st.write(df.head())
            temp_file_path = None
                
        if csv_file is not None:
            # Save the uploaded file to a temporary file
            temp_file_path = save_uploaded_file(csv_file)

            # Read CSV with explicit date format and ignore parsing errors
            df = read_csv_with_date_format(temp_file_path)

            # Display the uploaded CSV file
            st.subheader("Data:")
            st.write("This is what the first 5 rows of the data look like:")
            st.write(df.head())
            
        st.write("")
        st.write("")
        
        st.sidebar.subheader("Choose your input method:")
        input_method = st.sidebar.radio("Choose your Input", options=["Text", "Voice"])

        # Chat with GPT        
        if input_method == "Text":
            st.subheader("Visualize Data:")
            visualization_query = st.text_input("Enter a query for data visualization:")

            # Button to trigger the visualization action
            if st.button('Visualize'):
                if visualization_query:
                    prompt = f"CSV Data: {df.to_string(index=False)}\nVisualization Query: {visualization_query}"
                    visualization_response = openai.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "system", "content": "You are a world-class Data Analyst and your job is to provide only the code based on the user's query, nothing else other than the code and respective dataframe part. Only the code, do not add ``` and python in the code. Only give me the raw code and dataframe part. You will always use df to represent the dataframe"},
                                {"role": "user", "content": prompt}],temperature=0,max_tokens=300,)

                    try:
                        st.subheader("Output Data:")
                        prompt = f"Prompt response: {visualization_response.choices[0].message.content}"
                        visualization_response_ = openai.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "system", "content": "You are a world-class Data Analyst and your job is to filter column names based on the response, collect all column names in the response and create a string file separated by comma for all columns, do not add ``` and python in the code."},
                                    {"role": "user", "content": prompt}],temperature=0,max_tokens=200,)
                        
                        columns_list=visualization_response_.choices[0].message.content.split(', ')
                        temp_df=df[columns_list]
                        st.dataframe(temp_df)
                        st.write("")
                        st.write("")
                    except Exception as e:
                        pass
                    
                    try:
                        execute_generated_code(visualization_response.choices[0].message.content, df)
                    except Exception as e:
                        st.error("Unable to Load the plots")

        # Voice input method
        else:
            st.subheader("Visualize Data:")
            visualization_query_placeholder = st.empty()
            audio_bytes = audio_recorder(text = "Click to record", icon_size="2x", key="audio_button")
            st.markdown(""" <style> .body {<button type="button" class="btn btn-primary">Primary</button>}  </style>""",unsafe_allow_html=True)

            if audio_bytes:
                visualization_query=""
                try:
                    filename = "recorded_voice"+".wav"
                    with open(filename, mode='bx') as f:
                        f.write(audio_bytes)
                        sound = am.from_file(filename, format='wav', frame_rate=44100)
                        sound = sound.set_frame_rate(16000)
                        sound.export(filename, format='wav')
                        harvard = sr.AudioFile(filename)
                        with harvard as source:
                            audio = r.record(source)
                            visualization_query = r.recognize_google(audio)
                            st.write(f"You said: {visualization_query}")

                    if visualization_query:
                        prompt = f"CSV Data: {df.to_string(index=False)}\nVisualization Query: {visualization_query}"
                        visualization_response = openai.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "system", "content": "You are a world-class Data Analyst and your job is to provide only the code based on the user's query, nothing else other than the code. Only the code, do not add ``` and python in the code. Only give me the raw code. You will always use df to represent the dataframe"},
                                    {"role": "user", "content": prompt}],temperature=0.1,max_tokens=300,)

                        execute_generated_code(visualization_response.choices[0].message.content, df)
                except Exception as e:
                    pass
                os.remove(filename)
                
        # Remove the temporary file after use
        if temp_file_path is not None:
            os.remove(temp_file_path)


    if __name__ == "__main__":
        main()

if selected == "Data Scrubbing":
    st.title("Data Cleaning and Processing")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    chat = ChatOpenAI(temperature=0)
    chat = ChatOpenAI(temperature=0, openai_api_key="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")
    # if uploaded_file is not None:
    #     with st.spinner("Loading data..."):  # Display spinner while processing
    #         df = handle_uploaded_file(uploaded_file)
    # else:
    #     st.info("Please upload a CSV file to start.")
    #     df = pd.DataFrame()
    placeholder = st.empty()
    placeholder.text("Drag and drop your CSV file here, or click to browse.")

    # Progress bar for upload
    progress = st.progress(0)
    if uploaded_file is not None:
        progress.progress(20)  # Update progress on file selection

        # Read the CSV data with potential encoding and delimiter options
        df = pd.read_csv(
            uploaded_file,
            encoding="utf-8",  # Adjust encoding as needed
            delimiter=",",  # Adjust delimiter as needed
        )

        progress.progress(100)  # Update progress on completion

        # Display success message
        st.success("File uploaded successfully!")

        # Remove placeholder text
        placeholder.empty()

        # Display the DataFrame
        st.dataframe(df)
    
    user_input = st.text_input("Enter your text:", "Type here...")
    submit=st.button("Ask the question")
        
    
    
    messages = [
    SystemMessage(
        content="You are a expert data scientist, you have expertise in Data cleaning using pandas, you have to generate a code to clean the dataset df which is already loaded, give only code, give directly code"
    ),
    HumanMessage(
        content= user_input
    ),
]
    result = chat.invoke(messages)
    code_snippet = result.content.split('\n')[1]
    exec(code_snippet)
    
    if submit and input:
        result = chat.invoke(messages)
        code_snippet = result.content.split('\n')[1]
        st.write(code_snippet)
        if code_snippet[:3] == "df=":
            st.write(eval(code_snippet))
        else :
            df = eval(code_snippet)
        
            st.write(df)






