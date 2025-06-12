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
from auth import authenticate, logout_button

# Call login function
authenticate()

if st.session_state["authenticated"]:
    # Once authenticated, show logout and app
    logout_button()
    st.write(f"Welcome back, {st.session_state['username']}!")


    # Check if OPENAI_API_KEY is set
    api_key = st.secrets["OPENAI_API_KEY"]
    if not api_key:
        st.error("OPENAI_API_KEY not found in environment.")
    else:
        st.success("API key loaded successfully!")
    
    llm_config = LLMConfig(
        config_list=[
            {
                "model": "gpt-4.1",
                "api_key": api_key  # Pulls the API key from the environment
            }
        ]
    )
    search_tool = WebSearchPreviewTool(
        llm_config=llm_config,
        instructions="""Summarize web pages found""",
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
    print(f"Tools registered in agent: {search_agent.tools}")
    # File uploader
    uploaded_file = st.file_uploader("Upload your Excel file", type=["csv", "xlsx"])
    if uploaded_file and "summary_df" not in st.session_state:
        # Read the file into a DataFrame
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Display the uploaded table
        st.write("Uploaded Data:")
        st.dataframe(df)
        
        #Initialize an empty list to hold all results
        json_data_sukob = []

        eu_project_beneficiary = df[df["EU project beneficiary"] == "DA"].reset_index(drop=True)
        st.write(eu_project_beneficiary)
        associated_party = df[df["Associated party"] == "DA"].reset_index(drop=True)
        chunk_size = 5
        associated_party_chunks = [associated_party.iloc[i:i+chunk_size, :] for i in range(0, associated_party.shape[0], chunk_size)]
        print(f"Total chunks created: {len(associated_party_chunks)}")
        for i, chunk in enumerate(associated_party_chunks):
            print(f"Chunk {i+1}:\n{chunk}")
        global total_cost
        total_cost = 0
        global total_tokens
        total_tokens = 0
        global cost_descriptions
        cost_descriptions = []
        for (eu_project_beneficiary_name, eu_project_beneficiary_institution) in eu_project_beneficiary.loc[:, ["Name", "Institution"]].itertuples(index=False):
            for chunk in associated_party_chunks:
                task = f"Search online for websites mentioning EU project beneficiary '{eu_project_beneficiary_name}' from '{eu_project_beneficiary_institution}' and the following list of associated parties:"
                for (associated_party_name, associated_party_institution) in chunk.loc[:, ["Name", "Institution"]].itertuples(index=False):
                    task += f"\n - '{associated_party_name}' from '{associated_party_institution}'"
                task += "\n\n Each search should have only the names, not the institution. +\
                Only look for websites that mention both a person in group EU project beneficiary and a person in group Associated Party. +\
                Summarize the findings by calling the 'register_connection_entry' function.+\
                Even if you DO NOT find any websites that mention both names, register your finding of no articles mentioning both by calling the 'register_connection_entry' function.+\
                Findings should include both names and description of the connection between two individuals+\
                Findings should include a TRUE for potential_connection_found if any type of connection is found, however small and FALSE otherwise.+\
                Findings should include a separate TRUE for both_names_found if both names are mentioned in any context. FALSE otherwise.+\
                Findings for connection and both_names_found do not have to be the same. For instance, if both are employees at the same institution connection_found is TRUE +\
                However if both names were not found on at least one same web page both_names_found is FALSE."        
                st.write(task)        
                # Run the agent with the full query
                response = search_agent.run(
                    #recipient = search_agent,
                    message=task,
                    tools=search_agent.tools, 
                    user_input=False,
                    max_turns=3,
                )
                response.process()
                st.markdown(response.summary)
        summary_df = pd.DataFrame.from_records([x.model_dump() for x in potential_connection_found])
        st.write(summary_df)
        st.session_state.summary_df = summary_df
        output = BytesIO()
        if "summary_df" in st.session_state:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                summary_df.to_excel(writer, index=False, sheet_name='Results')
            output.seek(0)

            # Streamlit download button
            st.download_button(
                label="📥 Download Results as Excel",
                data=output,
                file_name="search_results_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.stop()  # Optional, defensive
