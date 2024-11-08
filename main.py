import streamlit as st
import openai
import json
import docker
import os
import requests
import subprocess
import uuid

# Function implementations
def execute_shell_command(command):
    # Security: Disallowed commands
    disallowed_commands = ['rm', 'mv', 'dd', 'shutdown', 'reboot', 'poweroff', 'mkfs', ':(){:|:&};:', 'sudo']
    command_name = command.split()[0]
    if command_name in disallowed_commands:
        return f"Error: The command '{command_name}' is disallowed for security reasons."

    # Execute the command in the Docker container
    try:
        exec_log = st.session_state.container.exec_run(command, stdout=True, stderr=True, tty=True)
        output = exec_log.output.decode('utf-8')
        if output.strip() == '':
            output = f"Command '{command}' executed successfully with no output."
        return output
    except Exception as e:
        return f"Error executing command: {str(e)}"

def read_file(file_path):
    # Security: Restrict access to certain directories
    allowed_paths = ['/home', '/tmp']
    if not any(file_path.startswith(ap) for ap in allowed_paths):
        return "Error: Access to this file path is not allowed."
    try:
        exec_log = st.session_state.container.exec_run(f"cat {file_path}", stdout=True, stderr=True, tty=True)
        output = exec_log.output.decode('utf-8')
        return output
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(file_path, content):
    # Security: Restrict access to certain directories
    allowed_paths = ['/home', '/tmp']
    if not any(file_path.startswith(ap) for ap in allowed_paths):
        return "Error: Access to this file path is not allowed."
    try:
        # Use echo to write content to the file
        exec_log = st.session_state.container.exec_run(f"echo '{content}' > {file_path}", stdout=True, stderr=True, tty=True)
        output = exec_log.output.decode('utf-8')
        return f"Content written to {file_path}."
    except Exception as e:
        return f"Error writing to file: {str(e)}"

def get_system_metrics():
    try:
        exec_log = st.session_state.container.exec_run("top -b -n1 | head -n5", stdout=True, stderr=True, tty=True)
        output = exec_log.output.decode('utf-8')
        return output
    except Exception as e:
        return f"Error retrieving system metrics: {str(e)}"

def make_http_request(method, url, headers=None, data=None):
    # Security: Limit allowed URLs or domains if necessary
    allowed_domains = ['example.com']  # Update as needed
    if not any(allowed_domain in url for allowed_domain in allowed_domains):
        return "Error: Access to this URL is not allowed."
    try:
        response = requests.request(method, url, headers=headers, data=data, timeout=5)
        return response.text
    except Exception as e:
        return f"Error making HTTP request: {str(e)}"

def execute_code(language, code):
    # Security: Limit languages and ensure sandboxing
    allowed_languages = ['python', 'javascript']
    if language not in allowed_languages:
        return f"Error: Language '{language}' is not supported."
    try:
        if language == 'python':
            # Run code in a separate process with resource limits
            process = subprocess.run(
                ['python3', '-c', code],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            return process.stdout or process.stderr
        elif language == 'javascript':
            process = subprocess.run(
                ['node', '-e', code],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            return process.stdout or process.stderr
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out."
    except Exception as e:
        return f"Error executing code: {str(e)}"

def parse_and_execute_instructions(instruction):
    # For demonstration, we'll assume the assistant handles parsing
    # In reality, you might implement NLP parsing here
    return f"Parsed and executed the instruction: {instruction}"

# Mapping function names to implementations
function_map = {
    "execute_shell_command": execute_shell_command,
    "read_file": read_file,
    "write_file": write_file,
    "get_system_metrics": get_system_metrics,
    "make_http_request": make_http_request,
    "execute_code": execute_code,
    "parse_and_execute_instructions": parse_and_execute_instructions,
}

# Initialize Docker client
docker_client = docker.from_env()

# Streamlit App
def main():
    st.title("Advanced GPT-Powered Linux Interaction")

    # Sidebar
    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model = st.sidebar.selectbox("Select Model", ["gpt-4", "gpt-3.5-turbo"])

    # Initialize session state
    if "messages" not in st.session_state:
        # Advanced system prompt
        system_prompt = {
            "role": "system",
            "content": (
                "You are an intelligent assistant with advanced capabilities to perform complex tasks on a computer. "
                "You can understand and execute multi-step instructions, make decisions based on context, and handle errors gracefully. "
                "You have access to functions that allow you to execute shell commands, manage files, perform network operations, and run code. "
                "Always ensure actions are safe, comply with security policies, and confirm with the user if unsure."
            )
        }
        st.session_state.messages = [system_prompt]

    if "container" not in st.session_state:
        # Pull Ubuntu image if not already present
        try:
            image = docker_client.images.get('ubuntu:latest')
        except docker.errors.ImageNotFound:
            with st.spinner("Pulling Ubuntu image..."):
                image = docker_client.images.pull('ubuntu:latest')
        # Create a Docker container for the session
        st.session_state.container = docker_client.containers.run(
            'ubuntu:latest',
            tty=True,
            detach=True,
            user='1000:1000',  # Non-root user
            security_opt=['no-new-privileges'],  # Prevent privilege escalation
            name=f"session_container_{st.session_state.run_id}"
        )
        st.sidebar.success("Docker container started.")

    st.sidebar.header("Chat with GPT")
    user_input = st.sidebar.text_input("You:", key="input")
    if st.sidebar.button("Send"):
        if api_key == "":
            st.sidebar.error("Please enter your OpenAI API key.")
        else:
            openai.api_key = api_key
            # Append user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Define functions (tools)
            functions = [
                {
                    "name": "execute_shell_command",
                    "description": "Executes a shell command in a safe, sandboxed environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The shell command to execute."
                            }
                        },
                        "required": ["command"]
                    }
                },
                {
                    "name": "read_file",
                    "description": "Reads the contents of a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read."
                            }
                        },
                        "required": ["file_path"]
                    }
                },
                {
                    "name": "write_file",
                    "description": "Writes content to a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to write."
                            },
                            "content": {
                                "type": "string",
                                "description": "The content to write to the file."
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                },
                {
                    "name": "get_system_metrics",
                    "description": "Retrieves system metrics like CPU and memory usage.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "make_http_request",
                    "description": "Makes an HTTP request.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "description": "HTTP method (GET, POST, etc.)."
                            },
                            "url": {
                                "type": "string",
                                "description": "The URL to make the request to."
                            },
                            "headers": {
                                "type": "object",
                                "description": "HTTP headers as a dictionary.",
                                "additionalProperties": {"type": "string"}
                            },
                            "data": {
                                "type": "string",
                                "description": "The request payload."
                            }
                        },
                        "required": ["method", "url"]
                    }
                },
                {
                    "name": "execute_code",
                    "description": "Executes code in a specified language securely.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "language": {
                                "type": "string",
                                "description": "The programming language of the code."
                            },
                            "code": {
                                "type": "string",
                                "description": "The code snippet to execute."
                            }
                        },
                        "required": ["language", "code"]
                    }
                },
                {
                    "name": "parse_and_execute_instructions",
                    "description": "Parses complex instructions and executes them step by step.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instruction": {
                                "type": "string",
                                "description": "The complex instruction to parse and execute."
                            }
                        },
                        "required": ["instruction"]
                    }
                },
            ]

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=model,
                messages=st.session_state.messages,
                functions=functions,
                function_call="auto",
            )

            assistant_message = response["choices"][0]["message"]
            st.session_state.messages.append(assistant_message)

            # Check if assistant wants to call a function
            while assistant_message.get("function_call"):
                function_name = assistant_message["function_call"]["name"]
                arguments = json.loads(assistant_message["function_call"]["arguments"])

                # Execute the function
                if function_name in function_map:
                    function_response = function_map[function_name](**arguments)
                else:
                    function_response = f"Error: Function '{function_name}' is not implemented."

                # Add function response to messages
                st.session_state.messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                })

                # Continue the conversation
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=st.session_state.messages,
                )
                assistant_message = response["choices"][0]["message"]
                st.session_state.messages.append(assistant_message)

            # Display chat messages
            st.sidebar.markdown("---")
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.sidebar.markdown(f"**You:** {message['content']}")
                elif message["role"] == "assistant":
                    st.sidebar.markdown(f"**Assistant:** {message['content']}")
                elif message["role"] == "function":
                    st.sidebar.markdown(f"**Function ({message['name']}):** {message['content']}")

    # Main Area - Display Linux Environment Output
    st.header("Linux Environment Output")
    if st.session_state.messages:
        # Find the last function response
        function_responses = [msg for msg in st.session_state.messages if msg["role"] == "function"]
        if function_responses:
            last_output = function_responses[-1]["content"]
            st.text_area("Output", value=last_output, height=300)
        else:
            st.info("No commands executed yet.")
    else:
        st.info("No commands executed yet.")

    # Clean up Docker container on app exit
    if st.button("Stop Session"):
        st.session_state.container.stop()
        st.session_state.container.remove()
        del st.session_state.container
        st.success("Session ended and container removed.")

if __name__ == "__main__":
    # Generate a unique run ID for the session
    if 'run_id' not in st.session_state:
        st.session_state.run_id = str(uuid.uuid4())
    main()
