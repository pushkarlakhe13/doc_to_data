from typing import List, Optional
import itertools
import requests

import pandas as pd
from pydantic import BaseModel, Field, validator
from kor import extract_from_documents, from_pydantic, create_extraction_chain
from kor.documents.html import MarkdownifyHTMLProcessor
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback

# It's better to do this an environment variable but putting it in plain text for clarity
openai_api_key = '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'

# Using gpt-3.5-turbo which is pretty cheap, but has worse quality
llm = ChatOpenAI(temperature=0,openai_api_key=openai_api_key)

from policy import Policy
from pydantic import BaseModel, Field, validator
from typing import Optional

# Generated Schema 
class Policy(BaseModel):
    scheme_name: str = Field(description="The full name of the scheme ")
    budget_allocation: str = Field(description="The budget allocated to the scheme")

schema, extraction_validator = from_pydantic(
    Policy,
    description="Information about the task given by the user",
    examples=[
        (
            '''Azadi Ka Amrit Mahotsav Mahila Samman Bachat Patra
            For commemorating Azadi Ka Amrit Mahotsav, a one-time new small
            savings scheme, Mahila Samman Savings Certificate, will be made available
            for a two-year period up to March 2025. This will offer deposit facility upto
            ` 2 lakh in the name of women or girls for a tenor of 2 years at fixed
            interest rate of 7.5 per cent with partial withdrawal option. ''',
            [
                {
"scheme_name": "Azadi Ka Amrit Mahotsav Mahila Samman Bachat Patra", "budget_allocation": "2 lakh"                }
            ],
        )
    ],
    many=True,
)

chain = create_extraction_chain(
    llm,
    schema,
    encoder_or_encoder_class="csv",
    validator=extraction_validator,
    input_formatter="triple_quotes",
)

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("budget_speech.pdf")
pages = loader.load_and_split()

with get_openai_callback() as cb:
    document_extraction_results = await extract_from_documents(
        chain, pages, max_concurrency=5, use_uid=False, return_exceptions=True
    )
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Successful Requests: {cb.successful_requests}")
    print(f"Total Cost (USD): ${cb.total_cost}")

validated_data = list(
    itertools.chain.from_iterable(
        extraction["validated_data"] for extraction in document_extraction_results
    )
)

scheme_df=pd.DataFrame(record.dict() for record in validated_data)

scheme_df




















