from pydantic import BaseModel, Field, create_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import logging

try:
    from noelle_settings.settings import OPENAI_API_KEY
except ImportError:
    # Fallback for demonstration if the settings file is not available
    import os
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("Warning: OPENAI_API_KEY not found. Please set it in your environment or settings file.")


logger = logging.getLogger(__name__)


def test_api_key():
    """
    Tests the OpenAI API key using LangChain's ChatOpenAI.
    """
    try:
        # 1. Initialize the LangChain ChatOpenAI model
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

        # 2. Invoke the model with a simple prompt.
        # The result is an AIMessage object; we access its content.
        response = llm.invoke("test")

        # 3. Return the text content
        return response.content
    except Exception as e:
        logger.error(f"OpenAI API key validation failed: {e}")
        return False


def generate_response(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    system_instruction_string: str = "Answer this prompt. You must provide a response in the requested JSON format.",
    response_schema_param: list = ["response"],
) -> str:
    """
    Generates a structured JSON response using an OpenAI model via LangChain.

    This method uses LangChain's structured output feature, which is powered by
    Pydantic models, to enforce the JSON schema.
    """
    try:
        # 1. Dynamically create a Pydantic model for the desired schema.
        # This is the standard LangChain way to define a JSON output structure.
        schema_fields = {
            param: (str, Field(..., description=f"The value for the '{param}' field."))
            for param in response_schema_param
        }
        DynamicResponseSchema = create_model(
            "DynamicResponseSchema",
            **schema_fields,
            __base__=BaseModel,
        )

        # 2. Initialize the LangChain ChatOpenAI Model
        llm = ChatOpenAI(model=model, openai_api_key=OPENAI_API_KEY, temperature=0)

        # 3. Chain the model with the structured output schema.
        # This tells the model to format its response according to our Pydantic model
        # and leverages OpenAI's native JSON mode under the hood.
        structured_llm = llm.with_structured_output(DynamicResponseSchema)

        # 4. Create the message list with system and user prompts
        messages = [
            SystemMessage(content=system_instruction_string),
            HumanMessage(content=prompt),
        ]

        # 5. Invoke the structured model
        response_model = structured_llm.invoke(messages)

        # 6. Convert the Pydantic model response to a JSON string to match
        # the original function's return type.
        return response_model.model_dump_json()

    except Exception as e:
        logger.error(f"An error occurred during LangChain OpenAI API call: {e}")
        raise
