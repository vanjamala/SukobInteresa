import streamlit as st
from typing import Literal
import os
from pydantic import BaseModel
from pydantic import Field
from typing import Annotated

from autogen import ConversableAgent, UserProxyAgent, LLMConfig
from autogen.tools.experimental.web_search_preview import WebSearchPreviewTool

from dotenv import load_dotenv
import pandas as pd
from io import BytesIO


load_dotenv()

# DEBUG: show the path of the current working directory
st.write("CWD:", os.getcwd())

# DEBUG: list files in the directory
st.write("Files in CWD:", os.listdir())

# Check if OPENAI_API_KEY is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment.")
else:
    st.success("API key loaded successfully!")
    
llm_config = LLMConfig(
    config_list=[
        {
            "model": "gpt-4.1",
            "api_key": os.getenv("OPENAI_API_KEY")  # Pulls the API key from the environment
        }
    ]
)
search_tool = WebSearchPreviewTool(
    llm_config=llm_config,
    instructions="""Summarize web pages found and print the summary on streamlit app using the registered function streamlit_summary_writer """,
)

   
instructions = """
Search for information using the web and create a summary of your findings. 
Once you find relevant results create a summary. 
"""

class ConnectionEntry(BaseModel):
    eu_project_beneficiary: Annotated[str, Field(description="Name of the EU project beneficiary.")]
    associated_party: Annotated[str, Field(description="Name of the associated party")]
    findings: str = Field(..., description="Description of the webpage in which names of both individuals were found or No connection found if nothing is found.")
    potential_connection_found: Annotated[bool, Field(description="A meaningful connection between individuals was found (True only if a meaningful connection between individuals was found. False otherwise.)")]
    both_names_found: Annotated[bool, Field(description="A specific viewport mentioning both individuals was found (True only if a webpage mentioning both individuals names is found. False otherwise.)")]
    links: Annotated[list[str], Field(description="Relevant links")]

potential_connection_found: list[ConnectionEntry] = []
def register_connection_entry(entry: Annotated[ConnectionEntry, "Entry for potential connection"]) -> str:
    print("Function execution started")  # Check if function is entered
    print(f"Received entry: {entry}")  # Debug message
    print(f"Type of entry: {type(entry)}")  # Debug the type of entry
    global potential_connection_found
    print(f"State of potential_connection_found before appending: {potential_connection_found}")  # Debug the list
    potential_connection_found.append(entry)
    # Now print the entire list after adding the new entry
    print(f"Potential connections found so far: {potential_connection_found}")  # This prints the entire list
    for entry in potential_connection_found:
      print(entry)
    return "Potential connection succesfully registred"
search_agent = ConversableAgent(
    name="SearchAssistant",
    system_message=instructions,
    llm_config=llm_config,
    functions=[register_connection_entry]
)
search_tool.register_for_llm(search_agent)

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["csv", "xlsx"])

# Button to run the query
#if st.button("Run Query") and user_query:
#    # Combine user query with hidden instructions
#    full_query = f"{user_query}\n{instructions}"

#    # Run the agent with the full query
#    response = search_agent.run(
#        #recipient = search_agent,
#        message=full_query,
#        tools=search_agent.tools, 
#        user_input=False,
#        max_turns=3,
#    )
#    response.process()
#    st.markdown(response.summary)

if uploaded_file:
    # Read the file into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Display the uploaded table
    st.write("Uploaded Data:")
    st.dataframe(df)
    
