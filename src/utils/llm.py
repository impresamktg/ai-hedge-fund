"""Helper functions for LLM"""

import json
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from utils.progress import progress

T = TypeVar('T', bound=BaseModel)

def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory = None
) -> T:
    """
    Makes an LLM call with retry logic, handling both Deepseek and non-Deepseek models.

    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """
    from llm.models import get_model, get_model_info

    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)

    # Handle different model requirements for structured output
    if model_info and model_info.provider == "OpenAI":
        schema = pydantic_model.schema()
        prompt_template = f"""You are a financial analysis assistant. Your response must be VALID JSON matching this schema:
        {json.dumps(schema, indent=2)}
        
        Respond ONLY with the JSON object, no other text. The JSON must be valid and match the schema exactly.
        
        Your task: {prompt}"""
        
        llm = ChatOpenAI(model=model_name, temperature=0)
        messages = [{"role": "system", "content": prompt_template}]
        
        for _ in range(3):  # Try up to 3 times
            try:
                response = llm.invoke(messages)
                content = response.content.strip()
                # Parse JSON from the response
                result = json.loads(content)
                # Validate against the model
                return pydantic_model.model_validate(result)
            except (json.JSONDecodeError, ValueError) as e:
                continue
        
        # If we get here, all attempts failed
        return pydantic_model.model_validate({})  # Return empty model
    else:
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

        # Call the LLM with retries
        for attempt in range(max_retries):
            try:
                # Call the LLM
                result = llm.invoke(prompt)

                # For non-JSON support models, we need to extract and parse the JSON manually
                if model_info and not model_info.has_json_mode():
                    parsed_result = extract_json_from_deepseek_response(result.content)
                    if parsed_result:
                        return pydantic_model(**parsed_result)
                else:
                    return result

            except Exception as e:
                if agent_name:
                    progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

                if attempt == max_retries - 1:
                    print(f"Error in LLM call after {max_retries} attempts: {e}")
                    # Use default_factory if provided, otherwise create a basic default
                    if default_factory:
                        return default_factory()
                    raise e

def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)

def extract_json_from_deepseek_response(content: str) -> Optional[dict]:
    """Extracts JSON from Deepseek's markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from Deepseek response: {e}")
    return None