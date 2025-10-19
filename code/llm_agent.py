import os
from typing import List, Dict
from gradio_client import Client
import requests
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display, update_display
import anthropic

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")


openai = OpenAI()
OPENAI_MODEL = 'gpt-4o-mini'

claude = anthropic.Anthropic()
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")


class QwenGradioClient:
    """Client for interacting with Qwen model hosted on Gradio/Colab"""
    
    def __init__(self, gradio_url):
        """
        Initialize the client
        
        Args:
            gradio_url: The Gradio public URL from your Colab notebook
                       Example: "https://abc123xyz.gradio.live"
        """
        print(f"Connecting to {gradio_url}...")
        try:
            self.client = Client(gradio_url)
            print("✓ Connected successfully!")
            print(f"✓ Model ready at: {gradio_url}")
            print("Available API endpoints:")
            print(self.client.view_api())
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            print("\nMake sure:")
            print("  1. Colab notebook is running")
            print("  2. Gradio URL is correct")
            print("  3. URL format: https://xxxxx.gradio.live")
    
    def generate(self, prompt, system_message="You are a helpful assistant.", 
                 max_tokens=2000, temperature=0.7, top_p=0.9):
        """
        Generate text from a prompt
        
        Args:
            prompt: The input prompt
            system_message: System message to set behavior
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.1 to 2.0)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text string
        """
        try:
            result = self.client.predict(
                prompt=prompt,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                api_name="/generate_text",
            )
            return result
        except Exception as e:
            return f"Error: {str(e)}"


# Initialize qwen client
GRADIO_URL = "https://8882ceafd3a70390bb.gradio.live"
qwen_client = QwenGradioClient(GRADIO_URL)


system_message = "You are a helpful assistant."

split_line = "================================================="


def _format_history_for_llama(history: List[Dict[str, str]], message: str) -> str:
    sections = [f"System: {system_message}"]
    for turn in history:
        role = turn.get("role", "user").capitalize()
        sections.append(f"{role}: {turn.get('content', '')}")
    sections.append(f"User: {message}")
    sections.append("Assistant:")
    return "\n\n".join(sections)


def _format_history_for_qwen(history: List[Dict[str, str]], message: str) -> str:
    conversation: List[str] = [f"<|im_start|>system\n{system_message}<|im_end|>"]
    role_map = {"system": "system", "user": "user", "assistant": "assistant"}

    for turn in history:
        role = role_map.get(turn.get("role", "user"), "user")
        content = turn.get("content", "")
        conversation.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    conversation.append(f"<|im_start|>user\n{message}<|im_end|>")
    conversation.append("<|im_start|>assistant\n")
    return "\n".join(conversation)


def _chat_with_llama(history: List[Dict[str, str]], message: str) -> str:
    prompt = _format_history_for_llama(history, message)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except requests.RequestException as exc:
        return f"Error calling local Llama model: {exc}"


def chat(message, history, llm_choice: str) -> str:
    normalized_choice = (llm_choice or "").lower()
    print(f"Using LLM choice: {normalized_choice}")

    if normalized_choice.startswith("gpt"):
        messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
        res = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages, stream=False)
        return res.choices[0].message.content

    if "llama" in normalized_choice:
        return _chat_with_llama(history, message)
    
    if normalized_choice.startswith("qwen"):
        print("Using Qwen model via Gradio client")
        formatted_prompt = _format_history_for_qwen(history, message)
        return qwen_client.generate(
            prompt=formatted_prompt,
            system_message=system_message,
            max_tokens=1200,
            temperature=0.1,
            top_p=0.9
        )

    # Default to Claude
    messages = history + [{"role": "user", "content": message}]
    res = claude.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1200,
        temperature=0.0,
        system=system_message,
        messages=messages,
    )
    return res.content[0].text


def generate_split_message(resume):
    message = """Here is my resume:\n\n"""
    message += resume
    message += f"""\n\nPlease split the resume into paragraphs, and split paragraphs by:{split_line}. \
Each paragraph should be one of the following: 1. A period of professional experience, 2. Whole educational experience, 3. Skill set, \
4. Publications, 5. Certificates. Don't change the content of the resume in paragraphs. For each paragraph, be sure to show at the beginning \
which category the paragraph belongs to. Please convert symbols like â€“ and â€¢ to readable symbols. Please don't add any starting and trailing \
remarks"""
    return message


def split_profile(example_resume, llm_choice):
    print("Splitting resume")
    split_message = generate_split_message(example_resume)
    split_result = chat(split_message, history=[], llm_choice=llm_choice)
    split_result_li = split_result.split(split_line)
    split_result_li = [r.strip() for r in split_result_li]
    return split_result_li


def evaluate_profile(input_text, resume_text, llm_choice):
    """Evaluate user's profile w.r.t. pasted job"""
    print("Evaluating resume w.r.t. pasted job")
    message = """Here is my resume:\n\n"""
    message += resume_text
    message += """\nHere's description of a job I'm interested in:\n\n"""
    message += input_text
    message += """\n\nPlease give a judgement on how well my profile matches the job description. The match should be given in this structure: \
For each requirement in the job description, show the requirement, and retrieve and show relevant experiences in my profile. For job requirements \
that are not in my profile, please list them. In the end of the judgement, please conclude with a score from 1 to 5, where 1 is the \
worst match and 5 is the best.
"""
    eval_res = chat(message, history=[], llm_choice=llm_choice)
    return eval_res


def generate_resume(input_text, resume_text, llm_choice):
    """Generate a tailored resume for pasted job"""
    print("Generating a tailored resume for pasted job")
    message = """Here is my resume:\n\n"""
    message += resume_text
    message += """\nHere's the description of a job I'm interested:\n\n"""
    message += input_text
    message += """\n\nPlease improve the writing of my resume based on the job description. During the improvement, \
be faithful to the original content. Please just output the improved resume and don't include any beginning or endding remarks."""
    gen_res = chat(message, history=[], llm_choice=llm_choice)
    return gen_res
