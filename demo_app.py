import streamlit as st
from typing import Tuple
from typing import List, Optional
import itertools
import pandas as pd
from pydantic import BaseModel, Field, validator
from kor import extract_from_documents, from_pydantic, create_extraction_chain
from kor.documents.html import MarkdownifyHTMLProcessor
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback
import asyncio


# Create a function to accept the PDF document as input:
# a. Add a file uploader UI component to your Streamlit app:

def get_pdf_file() -> bytes:
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        return file_bytes
    return None


#Create a function to accept a short natural language description of the data to be extracted:
#a. Add a text input UI component to your Streamlit app:

def get_description() -> str:
    description = st.text_input("Enter a short description of the data to be extracted")
    return description

#Create a function to accept user-defined IDs for schema fields:
#a. Add a text input UI component to your Streamlit app:
def get_field_ids() -> Tuple[str,str]:
    field_id_1 = st.text_input("Enter the first field ID ")
    field_id_2 = st.text_input("Enter the second field ID ")
    return field_id_1, field_id_2

# Create a function to accept an example text to be used in the schema 
# Add a text are UI component to your Streamlit app:

def get_example_text() -> str:
    example_text = st.text_area("Enter an example text to be used in the schema")
    return example_text

# Create a "Policy" class to accept user-defined IDs for schema fields,

def create_policy_class(field_id_1: str, field_id_2: str) -> BaseModel:
    class Policy(BaseModel):
        # Dynamically set the fields based on user input
        locals()[field_id_1]: str = Field(description=f"Field for {field_id_1}")
        locals()[field_id_2]: Optional[str] = Field(description=f"Field for {field_id_2}")

        @validator(field_id_1)
        def field_must_not_be_empty(cls, v):
            if not v:
                raise ValueError(f"{field_id_1} must not be empty")
            return v
    return Policy

#b. Update the schema creation to accept user input for description and example text:
def create_extraction_chain_and_validator(description: str, example_text: str, field_id_1: str, field_id_2: str):
    Policy = create_policy_class(field_id_1, field_id_2)
    
    schema, extraction_validator = from_pydantic(
        Policy,
        description=description,
        examples=[(example_text, [{field_id_1: "value_1", field_id_2: "value_2"}])],
        many=True,
    )
    return schema, extraction_validator

def get_openai_api_key():
    st.write("Please enter your OpenAI API Key:")
    openai_api_key = st.text_input("API Key")
    return openai_api_key

async def main():
    pdf_bytes = get_pdf_file()
    description = get_description()
    field_id_1, field_id_2 = get_field_ids()
    example_text = get_example_text()
    openai_api_key = get_openai_api_key()

    if pdf_bytes and description and field_id_1 and field_id_2 and example_text and openai_api_key:
        llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        schema, extraction_validator = create_extraction_chain_and_validator(description, example_text, field_id_1, field_id_2)
        chain = create_extraction_chain(llm, schema, encoder_or_encoder_class="csv", validator=extraction_validator, input_formatter="triple_quotes")

        loader = PyPDFLoader(pdf_bytes)
        pages = loader.load_and_split()

        with get_openai_callback() as cb:
            document_extraction_results = await extract_from_documents(chain, pages, max_concurrency=5, use_uid=False, return_exceptions=True)

        validated_data = list(
            itertools.chain.from_iterable(
                extraction["validated_data"] for extraction in document_extraction_results
            )
        )

        scheme_df = pd.DataFrame(record.dict() for record in validated_data)
        return scheme_df

    else:
        st.write("Please provide all required inputs.")


if __name__ == "__main__":
    extracted_data = asyncio.run(main())

    if extracted_data is not None:
        st.dataframe(extracted_data)

        csv = extracted_data.to_csv(index=False)
        st.download_button("Download CSV file", csv, file_name="schemes.csv")




   