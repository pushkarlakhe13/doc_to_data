# Importing the required packages
import streamlit as st
from policy import Policy

# Set the page title and icon for the app
st.set_page_config(page_title="Schema Auto Generator", page_icon=":robot:")
st.header("Generate Schema from user input")  # Display a header for the page

# Display a prompt for the user to upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", key="pdf_upload")

# Check if a file was uploaded and if the file is a PDF
if uploaded_file is not None:
    if uploaded_file.type == 'application/pdf':
        st.write("Upload successful!")
    else:
        st.write("Error: Please upload a PDF file.")

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
    if input_id is not None:
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
    schema_str += "    description=\"Information about the task given by the user\",\n"
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










