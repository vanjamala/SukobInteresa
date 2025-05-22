import streamlit as st
from typing import Literal
import os
from pydantic import BaseModel

from autogen import ConversableAgent, UserProxyAgent, LLMConfig
from autogen.tools.experimental.web_search_preview import WebSearchPreviewTool

from dotenv import load_dotenv
import pandas as pd
from io import BytesIO


load_dotenv()
llm_config = LLMConfig(
    config_list=[
        {
            "model": "gpt-4.1",
            "api_key": os.getenv("OPENAI_API_KEY")  # Pulls the API key from the environment
        }
    ]
)
user_proxy = ConversableAgent(
    name="UserProxy",
    llm_config=llm_config
)
search_tool = WebSearchPreviewTool(
    llm_config=llm_config,
    instructions="""Summarize web pages found and print the summary on streamlit app using the registered function streamlit_summary_writer """,
)
#search_tool.register_for_llm(user_proxy)

   
instructions = """
Search for information using the web and create a summary of your findings. 
Once you find relevant results create a summary. 
"""

search_agent = ConversableAgent(
    name="SearchAssistant",
    system_message=instructions,
    llm_config=llm_config,
    #functions=[streamlit_summary_writer]
)
search_tool.register_for_llm(search_agent)

print("Available tools for agent:")
for t in user_proxy.tools:
    print("-", t.name)

print("Function map:", search_agent._function_map)

# Input field for user query
user_query = st.text_input("Enter your query:")

# Button to run the query
if st.button("Run Query") and user_query:
    # Combine user query with hidden instructions
    full_query = f"{user_query}\n{instructions}"

    # Run the agent with the full query
    response = search_agent.run(
        #recipient = search_agent,
        message=full_query,
        tools=search_agent.tools, 
        user_input=False,
        max_turns=3,
    )
    response.process()
    st.markdown(response.summary)
    
