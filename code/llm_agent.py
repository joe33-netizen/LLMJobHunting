import os
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
MODEL = 'gpt-4o-mini'

claude = anthropic.Anthropic()

system_message = "You are a helpful assistant"


def chat(message, history, model="Claude"):
    # print("History is:")
    # print(history)
    # print("And messages is:")
    # print(messages)
    if model=="GPT":
        messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
        res = openai.chat.completions.create(model=MODEL, messages=messages, stream=False)
        response = res.choices[0].message.content
    else:
        messages = history + [{"role": "user", "content": message}]
        res = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            temperature=0., # deterministic
            system=system_message,
            messages=messages,
        )
        response = res.content[0].text
    return response


def generate_split_message(resume):
    split_line = "================================================="
    message = """Here is my resume:\n\n"""
    message += example_resume
    message += f"""\n\nPlease split the resume into paragraphs, and split paragraphs by:{split_line}. \
Each paragraph should be one of the following: 1. A period of professional experience, 2. Whole educational experience, 3. Skill set, \
4. Publications, 5. Certificates. Don't change the content of the resume in paragraphs. For each paragraph, be sure to show at the beginning \
which category the paragraph belongs to. Please convert symbols like â€“ and â€¢ to readable symbols. Please don't add any starting and trailing \
remarks"""
    return message


def split_profile(example_resume):
    print("Splitting resume")
    split_message = generate_split_message(example_resume)
    split_result = chat(split_message, history=[])
    split_result_li = split_result.split(split_line)
    split_result_li = [r.strip() for r in split_result_li]
    return split_result_li


def evaluate_profile(input_text, resume_text):
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
    eval_res = chat(message, history=[])
    return eval_res


def generate_resume(input_text, resume_text):
    """Generate a tailored resume for pasted job"""
    print("Generating a tailored resume for pasted job")
    message = """Here is my resume:\n\n"""
    message += resume_text
    message += """\nHere's the description of a job I'm interested:\n\n"""
    message += input_text
    message += """\n\nPlease improve the writing of my resume based on the job description. During the improvement, \
be faithful to the original content. Please just output the improved resume and don't include any beginning or endding remarks."""
    gen_res = chat(message, history=[])
    return gen_res
