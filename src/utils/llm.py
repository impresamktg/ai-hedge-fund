"""Helper functions for LLM"""

import json
import os
from typing import TypeVar, Type, Optional, Any, Callable
from pydantic import BaseModel
from utils.progress import progress
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

T = TypeVar('T', bound=BaseModel)

def call_llm(
    prompt: str | list,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[BaseModel],
    agent_name: str = None,
    default_factory: Callable = None,
) -> Any:
    print(f"\nDEBUG: LLM Call")
    print(f"DEBUG: Model Name = {model_name}")
    print(f"DEBUG: Model Provider = {model_provider}")
    print(f"DEBUG: Agent Name = {agent_name}")
    print(f"DEBUG: OpenAI API Key = {'Set' if os.getenv('OPENAI_API_KEY') else 'Not Set'}")
    print(f"DEBUG: Pydantic Model = {pydantic_model.__name__}")
    print(f"DEBUG: Prompt = {prompt}")  # Print the entire prompt
    from llm.models import get_model_info, ModelProvider
    from langchain_google_genai import ChatGoogleGenerativeAI

    model_info = get_model_info(model_name)
    try:
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [HumanMessage(content=str(prompt))]
        
        if model_info.provider == ModelProvider.GEMINI:
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
            response = llm.invoke(messages)
        elif model_info.provider == ModelProvider.OPENAI:
            llm = ChatOpenAI(model=model_name, temperature=0)
            response = llm.invoke(messages)
        else:
            print(f"DEBUG: Unsupported model provider: {model_info.provider}")
            return create_default_response(pydantic_model)
        print("DEBUG: LLM client initialized successfully")
    except Exception as e:
        print(f"DEBUG: Error initializing LLM client: {str(e)}")
        return create_default_response(pydantic_model)


    # Handle different model requirements for structured output
    if model_info and model_info.provider == "OpenAI":
        schema = pydantic_model.schema()
        prompt_template = f"""You are a financial analysis assistant. Your response must be VALID JSON matching this schema:
        {json.dumps(schema, indent=2)}

        Respond ONLY with the JSON object wrapped in triple backticks, no other text. The JSON must be valid and match the schema exactly.

        Your task: {prompt}"""

        # Assuming ChatOpenAI is defined elsewhere and handles messages correctly.
        try:
            llm = ChatOpenAI(model=model_name, temperature=0) # Assumed definition
            from langchain.schema import SystemMessage, HumanMessage
            if isinstance(prompt, list):
                messages = prompt  # Already formatted messages
            else:
                messages = [SystemMessage(content=prompt_template)]
            print("DEBUG: Created messages for LLM")
        except Exception as e:
            print(f"DEBUG: Error creating LLM messages: {str(e)}")
            return create_default_response(pydantic_model)

        for attempt in range(3):  # Try up to 3 times
            try:
                print(f"\nDEBUG: Attempt {attempt + 1}")
                print("\nDEBUG: Attempting LLM call...")
                response = llm.invoke(messages)
                print(f"\nDEBUG: Raw Response Content = {response.content}")
                print(f"DEBUG: Response Type = {type(response.content)}")

                # Handle the response content
                content = response.content
                if isinstance(content, str):
                    content = content.strip()
                    # Extract JSON from markdown code blocks if present
                    if "```" in content:
                        import re
                        json_match = re.search(r'```(?:json)?(.*?)```', content, re.DOTALL)
                        if json_match:
                            content = json_match.group(1).strip()
                    print(f"DEBUG: Cleaned Content = {content}")

                    try:
                        result = json.loads(content)
                        print(f"DEBUG: Parsed JSON = {result}")
                    except json.JSONDecodeError as je:
                        print(f"DEBUG: JSON Parse Error: {str(je)}")
                        print(f"DEBUG: Failed Content: {content}")
                        continue
                else:
                    result = response.content
                    print(f"DEBUG: Direct Content = {result}")

                try:
                    return pydantic_model.model_validate(result)
                except Exception as ve:
                    print(f"DEBUG: Validation Error: {str(ve)}")
                    print(f"DEBUG: Invalid Data: {result}")
                    continue

            except Exception as e:
                print(f"DEBUG: LLM Call Error: {str(e)}")
                print(f"DEBUG: Error Type: {type(e).__name__}")
                continue

        # If we get here, all attempts failed
        return pydantic_model.model_validate({})  # Return empty model
    else:
        try:
            if model_info.provider == ModelProvider.GEMINI:
                llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
            else:
                llm = ChatOpenAI(model=model_name, temperature=0)
                
            for attempt in range(3):
                try:
                    response = llm.invoke(messages)
                    content = response.content
                    
                    if isinstance(content, str):
                        if model_info and not model_info.has_json_mode():
                            try:
                                # Try to parse JSON directly first
                                json_data = json.loads(content)
                                return pydantic_model(**json_data)
                            except json.JSONDecodeError:
                                # If direct parsing fails, try to extract JSON from the response
                                import re
                                json_match = re.search(r'```(?:json)?(.*?)```', content, re.DOTALL)
                                if json_match:
                                    content = json_match.group(1).strip()
                                    return pydantic_model(**json.loads(content))
                    return content
                except Exception as e:
                    if agent_name:
                        progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/3")
                    if attempt == 2:
                        print(f"Error in LLM call after 3 attempts: {e}")
                        if default_factory:
                            return default_factory()
                        raise e
        except Exception as e:
            print(f"DEBUG: Error in non-OpenAI LLM handling: {str(e)}")
            return create_default_response(pydantic_model)



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