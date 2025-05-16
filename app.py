import gradio as gr
import os 
import time
from fastapi import FastAPI
from gradio import mount_gradio_app
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Configuration
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'  # Set in .env
height_correction = 200

# Only initialize Azure client if not in debug mode
if not DEBUG_MODE:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION", "2024-05-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

# Load prompt templates from files
def load_prompt_templates():
    templates = {}
    template_dir = os.path.join(os.path.dirname(__file__), "prompt_templates")
    for stage_num in range(1, 5):
        template_file = os.path.join(template_dir, f"stage_{stage_num}.txt")
        with open(template_file, "r") as file:
            templates[stage_num] = file.read()
    return templates

prompt_templates = load_prompt_templates()

def simulate_llm(formatted_prompt, corrections, template=None):
    """Mock LLM function for debug mode"""
    mock_response = f"""DEBUG MODE ACTIVE:
{template}

Context:
{formatted_prompt}

Corrections:
{corrections}

Generated Text: none
"""
    accumulated = ""
    for char in mock_response:
        accumulated += char
        time.sleep(0.02)  # Simulate delay
        yield accumulated

def azure_llm_call(formatted_prompt):
    """Actual Azure LLM call with streaming"""
    try:
        response = client.chat.completions.create(
            model=os.getenv("DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are a technical documentation expert."},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            stream=True
        )
        accumulated = ""
        for chunk in response:
            # Handle possible empty choices in chunk
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    accumulated += delta.content
                    yield accumulated
    except Exception as e:
        yield f"Error generating response: {str(e)}"

def get_llm_response(formatted_prompt, corrections, template):
    """Router function that handles debug/production mode"""
    if DEBUG_MODE:
        return simulate_llm(formatted_prompt, corrections, template)
    return azure_llm_call(formatted_prompt)

def get_stage_context(stage_num, stage_outputs):
    """Collect outputs from previous stages"""
    return "\n".join([f"Stage {i} Output:\n{stage_outputs[i]}" for i in range(1, stage_num) if stage_outputs[i]])

# Stage 1 Functions
def submit_correction_1(correction, chat):
    if correction.strip():
        chat.append((correction, ""))
    return chat

def generate_1(prompt, chat, stage_outputs):
    context = f"Initial Prompt: {prompt}"
    corrections = "\n".join([msg[0] for msg in chat])
    template = prompt_templates[1]
    formatted_prompt = template.format(context=context, corrections=corrections)
    llm_generator = get_llm_response(formatted_prompt, corrections, template)
    for chunk in llm_generator:
        yield chunk

def save_1(output_text, stage_outputs):
    stage_outputs[1] = output_text
    return stage_outputs

# Stage 2-4 Functions
def submit_correction_n(correction, chat):
    if correction.strip():
        chat.append((correction, ""))
    return chat

def generate_n(stage_num, chat, stage_outputs):
    context = get_stage_context(stage_num, stage_outputs)
    corrections = "\n".join([msg[0] for msg in chat])
    template = prompt_templates[stage_num]
    formatted_prompt = template.format(context=context, corrections=corrections)
    llm_generator = get_llm_response(formatted_prompt, corrections, template)
    for chunk in llm_generator:
        yield chunk

def save_n(stage_num, output_text, stage_outputs):
    stage_outputs[stage_num] = output_text
    return stage_outputs

# Gradio UI Setup
with gr.Blocks(title="LLM Wizard", theme=gr.themes.Soft()) as app:
    stage_outputs = gr.State({1: "", 2: "", 3: "", 4: ""})
    
    with gr.Tabs() as stages:
        # Stage 1
        with gr.Tab("Overview"):
            with gr.Row():
                gen_btn1 = gr.Button("Generate", scale=0)
                save_btn1 = gr.Button("Save", scale=0)
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Initial Prompt", lines=3)
                    correction1 = gr.Textbox(label="New Correction")
                    chat1 = gr.Chatbot(label="Corrections History", height=height_correction)
                output1 = gr.Textbox(label="Current Output", interactive=True, lines=15)
        
        # Stage 2
        with gr.Tab("Roles & Responsibilities"):
            with gr.Row():
                gen_btn2 = gr.Button("Generate", scale=0)
                save_btn2 = gr.Button("Save", scale=0)
            with gr.Row():
                with gr.Column():
                    chat2 = gr.Chatbot(label="Corrections History", height=height_correction)
                    correction2 = gr.Textbox(label="New Correction")
                output2 = gr.Textbox(label="Current Output", interactive=True, lines=15)
        
        # Stage 3
        with gr.Tab("Diagram"):
            with gr.Row():
                gen_btn3 = gr.Button("Generate", scale=0)
                save_btn3 = gr.Button("Save", scale=0)
            with gr.Row():
                with gr.Column():
                    chat3 = gr.Chatbot(label="Corrections History", height=height_correction)
                    correction3 = gr.Textbox(label="New Correction")
                output3 = gr.Textbox(label="Current Output", interactive=True, lines=15)
                mermaid_diagram = gr.Markdown(value="""
                    ```mermaid
                    graph LR
                        A --> B 
                        B --> C
                    ```
                    """)
        
        # Stage 4
        with gr.Tab("Full text"):
            with gr.Row():
                gen_btn4 = gr.Button("Generate", scale=0)
                save_btn4 = gr.Button("Save", scale=0)
            with gr.Row():
                with gr.Column():
                    chat4 = gr.Chatbot(label="Corrections History", height=height_correction)
                    correction4 = gr.Textbox(label="New Correction")
                output4 = gr.Textbox(label="Current Output", interactive=True, lines=15)

    # Event Handlers
    correction1.submit(submit_correction_1, [correction1, chat1], [chat1]).then(
        lambda: "", None, correction1)
    gen_btn1.click(generate_1, [prompt, chat1, stage_outputs], output1)
    save_btn1.click(save_1, [output1, stage_outputs], [stage_outputs])

    for i in range(2, 5):
        correction_box = globals()[f"correction{i}"]
        chat_box = globals()[f"chat{i}"]
        gen_btn = globals()[f"gen_btn{i}"]
        output_box = globals()[f"output{i}"]
        save_btn = globals()[f"save_btn{i}"]
        
        correction_box.submit(
            submit_correction_n, [correction_box, chat_box], [chat_box]
        ).then(lambda: "", None, correction_box)
        
        gen_btn.click(
            generate_n,
            inputs=[gr.State(i), chat_box, stage_outputs],
            outputs=output_box
        )

        save_btn.click(
            save_n,
            inputs=[gr.State(i), output_box, stage_outputs],
            outputs=stage_outputs
        )

# FastAPI integration
fastapi_app = FastAPI()
mount_gradio_app(fastapi_app, app, path="")

if __name__ == "__main__":
    app.launch()