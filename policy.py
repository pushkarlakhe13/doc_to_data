from pydantic import BaseModel, Field, validator

class Policy(BaseModel):
    input_id_1: str = Field(description="Descrption of input id 1 given by the user")
    input_id_2: str = Field(description="Descrption of input id 2 given by the user")

    @validator("input_id_1", pre=True, always=True)
    def input_id_1_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("Input Id 1 must not be empty")
        return v
