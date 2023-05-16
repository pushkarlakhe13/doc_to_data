from langchain.document_loaders import PyPDFLoader
import os

def process_pdf_file(user_uploaded_file):
    """
    This function takes a user uploaded file as input, loads it using PyPDFLoader and splits it into smaller chunks.
    """
    # Save user file to temp file
    with open('temp.pdf', 'wb') as f:
        f.write(user_uploaded_file.getbuffer())

    # Load and split the pdf file
    loader = PyPDFLoader('temp.pdf')
    pages = loader.load_and_split()

    # Remove the temporary file
    os.remove('temp.pdf')

    return pages

from pydantic import create_model
from typing import List, Tuple

def create_display_schema(task_description: str, input_ids: List[str], input_descs: List[str], example_input: str, expected_results: List[str]) -> str:
    # Start building the schema string
    schema_str = "class Policy(BaseModel):\n"

    # Add each input ID and description to the schema string
    for i, (input_id, input_desc) in enumerate(zip(input_ids, input_descs)):
        schema_str += f"    {input_id} : str = Field(description=\"{input_desc}\")\n"

    # Add the validator for the first input ID
    if input_ids:
        schema_str += f"\n    @validator(\"{input_ids[0]}\")\n"
        schema_str += f"    def {input_ids[0]}_must_not_be_empty(cls, v):\n"
        schema_str += f"        if not v:\n"
        schema_str += f"            raise ValueError(\"{input_ids[0]} must not be empty\")\n"
        schema_str += f"        return v\n"

    # Add the from_pydantic part of the schema
    schema_str += "\nschema, extraction_validator = from_pydantic(\n"
    schema_str += f"    Policy,\n"
    schema_str += f"    description=\"{task_description}\",\n"
    schema_str += "    examples=[\n"
    schema_str += "        (\n"
    schema_str += f"            '''\n{example_input}\n            ''',\n"
    schema_str += "            [\n"
    for i, (input_id, expected_result) in enumerate(zip(input_ids, expected_results)):
        schema_str += f"                {{\"{input_id}\": \"{expected_result}\"}},\n"
    schema_str += "            ],\n"
    schema_str += "        )\n"
    schema_str += "    ],\n"
    schema_str += "    many=True,\n"
    schema_str += ")\n"

    return schema_str

# Actual Schema generator 
from pydantic import create_model, Field, validator
from kor import from_pydantic

def create_extraction_schema(task_description, input_ids, input_descs, example_input, expected_results):
    # Dynamically create the Pydantic model
    fields = {input_id: (str, Field(description=input_desc)) for input_id, input_desc in zip(input_ids, input_descs)}
    
    # Create a validator for the first input_id
    fields[f"{input_ids[0]}_must_not_be_empty"] = validator(input_ids[0], allow_reuse=True)(lambda cls, v: v if v else ValueError(f"{input_ids[0]} must not be empty"))
    
    DynamicModel = create_model("DynamicModel", **fields)

    # Create the extraction schema
    schema, extraction_validator = from_pydantic(
        DynamicModel,
        description=task_description,
        examples=[
            (example_input, expected_results)
        ],
        many=True,
    )

    return schema, extraction_validator





# from pydantic import BaseModel, create_model, Field, validator
# from typing import List

# def generate_schema(input_ids: List[str], input_descs: List[str], task_description: str, example_input: str, expected_results: List[str]):
#     """
#     Generate a Pydantic schema based on user inputs
#     """
#     # Define a dictionary to hold our fields
#     fields = {}

#     # If input_id_1 is present, create a temporary base class with the validator
#     if "input_id_1" in input_ids:
#         class TempBase(BaseModel):
#             @validator("input_id_1")
#             def input_id_1_must_not_be_empty(cls, v):
#                 if not v:
#                     raise ValueError("Input Id 1 must not be empty")
#                 return v
#     else:
#         class TempBase(BaseModel):
#             pass

#     # Iterate over the input_ids and input_descs to create our fields
#     for input_id, input_desc in zip(input_ids, input_descs):
#         fields[input_id] = (str, Field(None, description=input_desc))

#     # Create a new Pydantic model with the fields
#     Policy = create_model('Policy', **fields, __base__=TempBase)

#     # We cannot add description, examples, and many fields to the model in the same way,
#     # because Pydantic does not support adding new attributes to a model.
#     # But we can return them separately and handle them in another way
#     description = task_description
#     examples = [(example_input, expected_results)]
#     return Policy, description, examples

# # Display schema genartor function 
# def display_schema(schema, task_description, example_input, expected_results):
#     """
#     Generate a string representation of the Pydantic schema
#     """
#     schema_str = f"class {schema.__name__}(BaseModel):\n"
#     for field_name, field in schema.__annotations__.items():
#         field_description = schema.__config__.get_field_by_name(field_name).field_info.description
#         schema_str += f"    {field_name}: {field.__name__} = Field(description=\"{field_description}\")\n"
        
#     if "input_id_1" in schema.__annotations__:
#         schema_str += "\n"
#         schema_str += "    @validator(\"input_id_1\")\n"
#         schema_str += "    def input_id_1_must_not_be_empty(cls, v):\n"
#         schema_str += "        if not v:\n"
#         schema_str += "            raise ValueError(\"Input Id 1 must not be empty\")\n"
#         schema_str += "        return v\n"

#     schema_str += "\n"
#     schema_str += "schema, extraction_validator = from_pydantic(\n"
#     schema_str += f"    {schema.__name__},\n"
#     schema_str += f"""    description=\"\"\"{task_description}\"\"\",\n"""
#     schema_str += "    examples=[\n"
#     schema_str += "        (\n"
#     schema_str += f"            '''{example_input}''',\n"
#     schema_str += "            [\n"
#     schema_str += "                {\n"
#     schema_str += ", ".join([f"\"{field_name}\": \"{expected_result}\"" for field_name, expected_result in zip(schema.__annotations__.keys(), expected_results)])
#     schema_str += "                }\n"
#     schema_str += "            ],\n"
#     schema_str += "        )\n"
#     schema_str += "    ],\n"
#     schema_str += "    many=True,\n"
#     schema_str += ")\n"

#     return schema_str
