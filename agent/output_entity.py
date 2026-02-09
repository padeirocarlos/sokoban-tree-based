from pydantic import BaseModel, Field

class GeneralResult(BaseModel):
    answers: str = Field(description="< The information regarding to final response>") 
    confidence: str = Field(description="< Here detailed classification of the confidence level of answer: High, Medium, or Low >")