from langchain.prompts import PromptTemplate
from typing import Optional

def get_prompt(
    template: Optional[str] = None
) -> PromptTemplate:
    # Use default template if none provided
    default_template = (
        "Based on this context: {context} Answer the question: {question}. "
        "If you do not know the answer, just say that you do not know. "
        "Do not try to make up an answer."
    )
    prompt_template = template or default_template

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"], 
    )

    return prompt