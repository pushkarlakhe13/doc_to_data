# Importing the required packages
import streamlit as st
from typing import List
from typing import List, Optional
import itertools
import requests
import pandas as pd
import pydantic 
from pydantic import BaseModel, Field, validator
from kor import extract_from_documents, from_pydantic, create_extraction_chain

from kor.documents.html import MarkdownifyHTMLProcessor
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import PyPDFLoader

import os
import tempfile


# Set the page title and icon for the app
st.set_page_config(page_title="Schema Auto Generator", page_icon=":robot:")
st.header(" Generate Schema from user input")  # Display a header for the page



col1, col2 = st.columns(2)
with col1 : 
    st.write('''
    Structured Data Extractor is a Streamlit-based 
    web application designed to help users to extract structured data 
    from unstructred documents like PDFs. It creates data extraction
    schemas from user inputs. The application allows users 
    to define input fields, their descriptions, and the expected results for each input. 
    he generated schema is then passed to 
    GPT models to extract relevant information
    ''')
with col2 :
    st.image("DryvIQ-Unstructured-VS-Structured-Data-Diagram-Light.png",width = 500)

# # Define a function to process the PDF file
# def process_pdf_file(uploaded_file):
#     if uploaded_file.type == 'application/pdf':
#         st.write("Upload successful!")
        
#         # Create a temporary file to store the uploaded PDF
#         with tempfile.NamedTemporaryFile(delete=False) as f:
#             f.write(uploaded_file.read())
#             pdf_file_path = f.name
        
#         # Create PyPDFLoader object and load PDF file
#         loader = PyPDFLoader(pdf_file_path)
#         pages = loader.load_and_split()
        
#         # Display information about the uploaded file
#         st.write("File Name:", uploaded_file.name)
#         st.write("File Size:", uploaded_file.size, "bytes")
#         st.write("Number of Pages:", len(pages))
        
#         # Process the PDF file
#         # ...
        
#         # Display the results
#         # ...
        
#         # Remove the temporary file
#         os.remove(pdf_file_path)
        
#     else:
#         st.write("Error: Please upload a PDF file.")
        
# # Display a prompt for the user to upload a PDF file
# uploaded_file = st.file_uploader("Upload a PDF file", key="pdf_upload")

# Display a prompt for the user to upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", key="pdf_upload")

# Check if a file was uploaded and if the file is a PDF
if uploaded_file is not None:
    if uploaded_file.type == 'application/pdf':
        st.write("Upload successful!")
    else:
        st.write("Error: Please upload a PDF file.")


# Check if a file was uploaded and if the file is a PDF
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.write("File Name:", uploaded_file.name)
        st.write("File Size:", uploaded_file.size, "bytes")
    with col2:
        st.write("Number of Pages: Not Processed Yet")
        
    if st.button("Process"):
        process_pdf_file(uploaded_file)

# Display a prompt for the user to enter a short description of the data extraction task
st.write(" ## Enter a short description of the data extraction task")
task_description = st.text_area("", key="task_description")  # Create a text area for user input

# Display a prompt for the user to enter the field IDs and descriptions for the data to be extracted
st.write(" ## Enter the field IDs and descriptions for the data to be extracted")

# Create a dropdown menu for selecting the number of input IDs
num_options = list(range(1, 11))
num_inputs = st.selectbox("Number of input IDs", options=num_options, format_func=str)

col1, col2 = st.columns(2)

# Create empty lists to store input IDs and descriptions
input_ids = []
input_descs = []

# Loop through the number of inputs selected by the user
for i in range(num_inputs):
    with col1:  # Create a text input field for the user to enter the input ID, and store the input in the input_id variable
        input_id = st.text_input(f"Enter input ID {i + 1}", key=f"input_id_{i + 1}")
    with col2:  # Create a text input field for the user to enter the description for the input ID, and store the input in the input_desc variable
        input_desc = st.text_input(f"Enter description for input ID {i + 1}", key=f"input_desc_{i + 1}")
    # If the input ID is not empty, append the input ID and description to their respective lists, and display the input ID and description back to the user
    if input_id:
        input_ids.append(input_id)
        input_descs.append(input_desc)
        st.write(f"You entered: {input_id} - {input_desc}")
    
st.write(" ## Enter an example for the model to understand", key="example_input")
example_input = st.text_area("")

st.write("## Input IDs and their expected results")
col3, col4 = st.columns(2)
expected_results = []

for i, input_id in enumerate(input_ids):
    with col3:
        st.write(f"Input ID {i + 1}: {input_id}")
    with col4:
        expected_result = st.text_input(f"Enter expected result for input ID {i + 1}", key=f"expected_result_{i + 1}")
        expected_results.append(expected_result)

with st.expander("See generated Schema"):
        st.write("## Generated Schema")
        schema_str = f"class Policy(BaseModel):\n"
        for input_id, input_desc in zip(input_ids, input_descs):
            schema_str += f"    {input_id}: str = Field(description=\"{input_desc}\")\n"
        
        if "input_id_1" in input_ids:
            schema_str += "\n"
            schema_str += "    @validator(\"input_id_1\")\n"
            schema_str += "    def input_id_1_must_not_be_empty(cls, v):\n"
            schema_str += "        if not v:\n"
            schema_str += "            raise ValueError(\"Input Id 1 must not be empty\")\n"
            schema_str += "        return v\n"

        schema_str += "\n"
        schema_str += "schema, extraction_validator = from_pydantic(\n"
        schema_str += "    Policy,\n"
        schema_str += f"""    description=\"\"\"{task_description}\"\"\",\n"""
        schema_str += "    examples=[\n"
        schema_str += "        (\n"
        schema_str += f"            '''{example_input}''',\n"
        schema_str += "            [\n"
        schema_str += "                {\n"
        schema_str += ", ".join([f"\"{input_id}\": \"{expected_result}\"" for input_id, expected_result in zip(input_ids, expected_results)])
        schema_str += "                }\n"
        schema_str += "            ],\n"
        schema_str += "        )\n"
        schema_str += "    ],\n"
        schema_str += "    many=True,\n"
        schema_str += ")\n"

        st.code(schema_str)


    

# # # Define openai_api_key as None initially
# openai_api_key = None

# # # Define a function to accept the OpenAI API key from the user and store it as an environment variable
# def accept_openai_api_key():
#     global openai_api_key  # Use the global keyword to access the global openai_api_key variable
#     openai_api_key = st.sidebar.text_input("OpenAI API Key")
#     if openai_api_key:
#         os.environ["OPENAI_API_KEY"] = openai_api_key

# # Call the accept_openai_api_key() function to get the OpenAI API key from the user and store it as an environment variable
# accept_openai_api_key()

# # Check if openai_api_key has been defined before creating llm object
# if openai_api_key is not None:
#     llm = ChatOpenAI(temperature=0,openai_api_key=openai_api_key)
#     # Generate schema
#     exec(schema_str)
#     generate_schema_str = exec(schema_str)
#     # Create extraction chain
#     chain = create_extraction_chain(
#         llm,
#         generate_schema_str,
#         encoder_or_encoder_class="csv",
#         validator=extraction_validator,
#         input_formatter="triple_quotes",
#     )

# Accept OpenAI API Key 

open_ai_api_key = st.text_input("Enter your OpenAI API Key", key= 'open_ai_api_key')

# schema = schema_str

# # Specifying the large language model LLM 

# llm = llm = ChatOpenAI(temperature=0,openai_api_key=open_ai_api_key)

# # Defining a function to run chain 
# def extraction_process() : 
#     chain= create_extraction_chain(
#         llm,
#         schema
#         encoder_or_encoder_class = 'csv'
#         validator = extraction_validator 
#         input_formatter='triple_quotes'
#     )
#     return chain 

def extraction_process(llm, schema, encoder_or_encoder_class, validator, input_formatter):
    chain = create_extraction_chain(
        llm,
        schema,
        encoder_or_encoder_class=encoder_or_encoder_class,
        validator=validator,
        input_formatter=input_formatter
    )
    return chain


st.button("Extract Data")


# # Run the chain 
# chain = create_extraction_chain(
#     llm,
#     schema,
#     encoder_or_encoder_class="csv",
#     validator=extraction_validator,
#     input_formatter="triple_quotes",
# )
#
