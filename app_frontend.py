
# Required Libraries 
import streamlit as st
from typing import List
from langchain.document_loaders import PyPDFLoader
import os

from pydantic import create_model
from typing import List, Tuple
from pydantic import create_model, Field, validator
from kor import from_pydantic

# Import app backend functions
from app_backend  import process_pdf_file 
from app_backend import create_display_schema


# Set the page title and icon for the app
st.set_page_config(page_title="Schema Auto Generator", page_icon=":robot:")

# Page layout
st.title("Schema Auto Generator")
st.markdown(""" 
This application is designed to help users extract structured data 
from unstructured documents like PDFs. It creates data extraction
schemas from user inputs and then uses a GPT model to extract relevant information.
""")

# Image
#st.image("DryvIQ-Unstructured-VS-Structured-Data-Diagram-Light.png", use_column_width=True, caption="Structured VS Unstructured Data")

# Upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

# Check if a file was uploaded and if the file is a PDF
if uploaded_file is not None:
    st.success("File upload successful!")
    st.write("File Name:", uploaded_file.name)
    st.write("File Size:", uploaded_file.size, "bytes")
else:
    st.info("Please upload a PDF file")

# Process button
if uploaded_file is not None:
    if st.button("Process"):
        pages = process_pdf_file(uploaded_file)
        st.write("Number of Pages:", len(pages))

# Task description
task_description = st.text_area("Enter a short description of the data extraction task", value="")

# Field IDs and descriptions
st.header("Field IDs and Descriptions")
num_inputs = st.number_input("Number of input IDs", min_value=1, max_value=10, value=1, step=1)

# Create empty lists to store input IDs and descriptions
input_ids = []
input_descs = []

# Loop through the number of inputs selected by the user
for i in range(num_inputs):
    # Create a new row with two columns
    cols = st.columns(2)
    
    # In the first column, ask for the input ID
    input_id = cols[0].text_input(f"Enter input ID {i + 1}", key=f"input_id_{i + 1}")
    
    # In the second column, ask for the description of the input ID
    input_desc = cols[1].text_input(f"Enter description for input ID {i + 1}", key=f"input_desc_{i + 1}")
    
    # If the input ID is not empty, append the input ID and description to their respective lists
    if input_id:
        input_ids.append(input_id)
        input_descs.append(input_desc)

# Example for the model
st.header("Example for the Model")
example_input = st.text_area("", key="example_input")

# Expected results
st.header("Expected Results")
expected_results = []
for i, input_id in enumerate(input_ids):
    expected_result = st.text_input(f"Enter expected result for input ID {i + 1}", key=f"expected_result_{i + 1}")
    expected_results.append(expected_result)

if st.button("Generate Schema"):
    # Call the backend function to generate the display schema
    schema_str = create_display_schema(task_description, input_ids, input_descs, example_input, expected_results)

    # Display the schema to the user
    st.code(schema_str, language='python')

# # Generate schema button
# if st.button("Generate Schema"):
#     # Generate the Pydantic schema
#     schema, description, examples = generate_schema(input_ids, input_descs, task_description, example_input, expected_results)

#     # Display the generated schema
#     schema_str = display_schema(schema, description, example_input, expected_results)

# # Show generated schema
# st.header("Generated Schema")
# if schema_str:
#     st.code(schema_str, language='python')
# else:
#     st.info("Please generate a schema by filling the required information and clicking the 'Generate Schema' button.")

# # Extract data button
# if st.button("Extract Data"):
#     # Call backend function to extract data
#     # TODO: Replace with your actual function
#     extracted_data = extract_data(uploaded_file, schema_str)

# # Show extracted data
# st.header("Extracted Data")
# if extracted_data is not None:
#     st.dataframe(extracted_data)
# else:
#     st.info("Please extract data by clicking the 'Extract Data' button.")

# # Download data button
# if extracted_data is not None:
#     csv = extracted_data.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
#     href = f'<a href="data:file/csv;base64,{b64}" download="extracted_data.csv">Download Extracted Data as CSV</a>'
#     st.markdown(href, unsafe_allow_html=True)
