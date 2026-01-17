import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List
import litellm
from dotenv import load_dotenv

# Disable pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
os.environ["LITELLM_IGNORE_PYDANTIC_WARNINGS"] = "1"

# Silence litellm logs
logging.getLogger("litellm").setLevel(logging.WARNING)
os.environ["LITELLM_LOG"] = "ERROR"

load_dotenv()

# Build litellm config
llm_config = {}
if os.environ.get("API_BASE"):
    llm_config["api_base"] = os.environ["API_BASE"]


# Terminal colors for output
YOU_COLOR = "\u001b[94m"          # Blue
ASSISTANT_COLOR = "\u001b[93m"    # Yellow
TOOL_COLOR = "\u001b[92m"         # Green
ERROR_COLOR = "\u001b[91m"        # Red
SUCCESS_COLOR = "\u001b[92m"      # Green
INFO_COLOR = "\u001b[96m"         # Cyan
RESET_COLOR = "\u001b[0m"

# Icons for better visual feedback
TOOL_ICON = "🔧"
FILE_ICON = "📄"
DIR_ICON = "📁"
SUCCESS_ICON = "✅"
ERROR_ICON = "❌"
THINKING_ICON = "🤔"


def resolve_abs_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def read_file_tool(filename: str) -> Dict[str, Any]:
    full_path = resolve_abs_path(filename)
    print(f"{TOOL_COLOR}{TOOL_ICON} Reading file: {INFO_COLOR}{filename}{RESET_COLOR}")
    with open(str(full_path), "r") as f:
        content = f.read()
    return {"file_path": str(full_path), "content": content}


def list_files_tool(path: str) -> Dict[str, Any]:
    full_path = resolve_abs_path(path)
    print(f"{TOOL_COLOR}{TOOL_ICON} Listing directory: {INFO_COLOR}{full_path}{RESET_COLOR}")
    all_files = []
    for item in full_path.iterdir():
        icon = FILE_ICON if item.is_file() else DIR_ICON
        all_files.append(
            {"filename": item.name, "type": "file" if item.is_file() else "dir"}
        )
    return {"path": str(full_path), "files": all_files}


def edit_file_tool(path: str, old_str: str, new_str: str) -> Dict[str, Any]:
    full_path = resolve_abs_path(path)
    if old_str == "":
        print(f"{TOOL_COLOR}{TOOL_ICON} Creating file: {INFO_COLOR}{path}{RESET_COLOR}")
        full_path.write_text(new_str, encoding="utf-8")
        print(f"{SUCCESS_COLOR}{SUCCESS_ICON} File created successfully{RESET_COLOR}")
        return {"path": str(full_path), "action": "created_file"}
    original = full_path.read_text(encoding="utf-8")
    if original.find(old_str) == -1:
        print(f"{ERROR_COLOR}{ERROR_ICON} Text to replace not found in file{RESET_COLOR}")
        return {"path": str(full_path), "action": "old_str not found"}
    print(f"{TOOL_COLOR}{TOOL_ICON} Editing file: {INFO_COLOR}{path}{RESET_COLOR}")
    edited = original.replace(old_str, new_str, 1)
    full_path.write_text(edited, encoding="utf-8")
    print(f"{SUCCESS_COLOR}{SUCCESS_ICON} File edited successfully{RESET_COLOR}")
    return {"path": str(full_path), "action": "edited"}


# Define tools in OpenAI format for litellm
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Gets the full content of a file provided by the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to read."
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Lists the files in a directory provided by the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to a directory to list files from."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replaces first occurrence of old_str with new_str in file. If old_str is empty, create/overwrite file with new_str.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to edit."
                    },
                    "old_str": {
                        "type": "string",
                        "description": "The string to replace."
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The string to replace with."
                    }
                },
                "required": ["path", "old_str", "new_str"]
            }
        }
    }
]

TOOL_REGISTRY = {
    "read_file": read_file_tool,
    "list_files": list_files_tool,
    "edit_file": edit_file_tool,
}

SYSTEM_PROMPT = """
You are a coding assistant whose goal it is to help us solve coding tasks.
You have access to tools for reading files, listing directories, and editing files.
Use these tools when needed to help with coding tasks.
"""





def llm_completion(conversation: List[Dict[str, str]]):
    # Prepare messages for litellm
    messages = []
    for msg in conversation:
        if msg["role"] == "system":
            messages.append({"role": "system", "content": msg["content"]})
        else:
            messages.append(msg)

    # Use litellm for provider-agnostic LLM calls
    model = os.environ["MODEL"]
    api_key = os.environ["API_KEY"]

    # Build kwargs for litellm
    kwargs = {
        "model": model,
        "api_key": api_key,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.1,
        "tools": TOOLS,
    }

    # Add api_base if available
    if llm_config.get("api_base"):
        kwargs["api_base"] = llm_config["api_base"]

    try:
        response = litellm.completion(**kwargs)
        return response
    except Exception as e:
        error_msg = f"LLM call failed: {str(e)}"
        print(f"{ERROR_COLOR}{ERROR_ICON} {error_msg}{RESET_COLOR}")
        print(f"{INFO_COLOR}Make sure you have set up your API keys in the .env file{RESET_COLOR}")
        print(f"{INFO_COLOR}Current model: {model}{RESET_COLOR}")
        return f"I encountered an error: {error_msg}. Please check your API key configuration."


def agent_loop():
    print(f"{SUCCESS_COLOR}{SUCCESS_ICON} Starting coding agent with litellm (provider-agnostic)...{RESET_COLOR}")
    print(f"{INFO_COLOR}Type 'exit' or press Ctrl+C to quit.{RESET_COLOR}\n")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input(f"{YOU_COLOR}You:{RESET_COLOR} ")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{INFO_COLOR}Goodbye! 👋{RESET_COLOR}")
            break

        if user_input.lower() in ["exit", "quit"]:
            print(f"{INFO_COLOR}Goodbye! 👋{RESET_COLOR}")
            break

        conversation.append({"role": "user", "content": user_input.strip()})

        # Show thinking indicator
        print(f"{THINKING_ICON} Assistant is thinking...", end="\r")

        while True:
            response = llm_completion(conversation)

            # Clear the thinking indicator
            print(" " * 30, end="\r")

            # Handle error responses
            if isinstance(response, str):
                print(f"{ASSISTANT_COLOR}Assistant:{RESET_COLOR} {response}")
                break

            try:
                # Get the assistant's message - handle different response formats
                if hasattr(response, 'choices') and response.choices:
                    assistant_message = response.choices[0].message
                    content = getattr(assistant_message, 'content', '') or ""

                    # Check if there are tool calls
                    tool_calls = getattr(assistant_message, 'tool_calls', None) or []

                    if not tool_calls:
                        # No tool calls, just print the response
                        if content.strip():
                            print(f"{ASSISTANT_COLOR}Assistant:{RESET_COLOR} {content}")
                        conversation.append({"role": "assistant", "content": content})
                        break

                    # Show assistant's initial response if any
                    if content.strip():
                        print(f"{ASSISTANT_COLOR}Assistant:{RESET_COLOR} {content}")

                    # Handle tool calls with better formatting
                    conversation.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls
                    })

                    print(f"{TOOL_COLOR}🔄 Executing {len(tool_calls)} tool{'s' if len(tool_calls) > 1 else ''}...{RESET_COLOR}")

                    for i, tool_call in enumerate(tool_calls, 1):
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        # Format tool call display
                        args_display = ", ".join(f"{k}={v}" for k, v in tool_args.items())
                        print(f"  {i}. {TOOL_ICON} {tool_name}({args_display})")

                        tool = TOOL_REGISTRY.get(tool_name)
                        if not tool:
                            error_msg = f"Unknown tool: {tool_name}"
                            print(f"     {ERROR_COLOR}{ERROR_ICON} {error_msg}{RESET_COLOR}")
                            conversation.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps({"error": error_msg})
                            })
                            continue

                        try:
                            resp = tool(**tool_args)
                            conversation.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(resp)
                            })
                        except Exception as e:
                            error_msg = f"Tool execution failed: {str(e)}"
                            print(f"     {ERROR_COLOR}{ERROR_ICON} {error_msg}{RESET_COLOR}")
                            conversation.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps({"error": error_msg})
                            })
                else:
                    # Fallback for unexpected response format
                    content = str(response)
                    print(f"{ASSISTANT_COLOR}Assistant:{RESET_COLOR} {content}")
                    conversation.append({"role": "assistant", "content": content})
                    break

            except Exception as e:
                print(f"{ERROR_COLOR}{ERROR_ICON} Error processing response: {e}{RESET_COLOR}")
                print(f"{ASSISTANT_COLOR}Assistant:{RESET_COLOR} I encountered an error processing the response.")
                break


if __name__ == "__main__":
    agent_loop()
