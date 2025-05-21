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
print(f"DEBUG_MODE: {DEBUG_MODE} (raw: {os.getenv('DEBUG_MODE')})")

DEBUG_STREAMING = os.getenv('DEBUG_STREAMING', 'True').lower() == 'true'  # Set in .env
print(f"DEBUG_STREAMING: {DEBUG_STREAMING} (raw: {os.getenv('DEBUG_STREAMING')})")

height_correction = 200

# Parse available models from environment variable
DEPLOYMENT_NAMES = os.getenv("DEPLOYMENT_NAMES").split(",")
DEPLOYMENT_NAMES = [name.strip() for name in DEPLOYMENT_NAMES if name.strip()]

# Mask model names with planet names
PLANET_NAMES = [
    "Mercury", "Venus", "Earth", "Mars", "Jupiter",
    "Saturn", "Uranus", "Neptune", "Pluto", "Ceres",
    "Eris", "Haumea", "Makemake"
]
# Map planet names to deployment names
planet_to_deployment = {}
deployment_to_planet = {}
for i, deployment in enumerate(DEPLOYMENT_NAMES):
    planet = PLANET_NAMES[i % len(PLANET_NAMES)]
    planet_to_deployment[planet] = deployment
    deployment_to_planet[deployment] = planet

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
    #{template}
    # Corrections to be made:
    # {corrections}
    # Context:
    mock_response = f"""DEBUG MODE ACTIVE:


{formatted_prompt}

Generated Text: none
"""
    if DEBUG_STREAMING:
        accumulated = ""
        for char in mock_response:
            accumulated += char
            time.sleep(0.02)  # Simulate delay
            yield accumulated
    else:
        yield mock_response

def filter_deepseek_thinking(text):
    """
    Removes content between <think> and </think> tags for DeepSeek R1 outputs.
    If <think> is found, removes it and everything up to and including </think>.
    """
    start_tag = "<think>"
    end_tag = "</think>"
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag, start_idx + len(start_tag))
    if start_idx != -1 and end_idx != -1:
        # Remove the <think>...</think> block
        filtered = text[:start_idx] + text[end_idx + len(end_tag):]
        return filtered.strip()
    return text

def azure_llm_call(formatted_prompt, deployment_name):
    """Actual Azure LLM call with streaming"""
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a technical documentation expert."},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            stream=True
        )
        if deployment_name.lower().startswith("deepseek"):
            # Buffer output until all <think>...</think> is gone, then stream
            buffer = ""
            thinking_removed = False
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        buffer += delta.content
                        if not thinking_removed:
                            # Remove <think>...</think> if present
                            filtered = filter_deepseek_thinking(buffer)
                            if filtered != buffer:
                                # <think>...</think> was present and removed
                                buffer = filtered
                                if buffer:
                                    thinking_removed = True
                                    yield buffer
                            elif "<think>" not in buffer:
                                # No <think> tag at all, start streaming
                                thinking_removed = True
                                if buffer:
                                    yield buffer
                        else:
                            yield buffer
        else:
            accumulated = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        accumulated += delta.content
                        yield accumulated
    except Exception as e:
        yield f"Error generating response: {str(e)}"

def get_llm_response(formatted_prompt, corrections, template, deployment_name):
    """Router function that handles debug/production mode"""
    if DEBUG_MODE:
        return simulate_llm(formatted_prompt, corrections, template)
    return azure_llm_call(formatted_prompt, deployment_name)

def get_stage_context(stage_num, stage_outputs):
    """Collect outputs from previous stages"""
    return "\n".join([f"Stage {i} Output:\n{stage_outputs[i]}" for i in range(1, stage_num) if stage_outputs[i]])

# --- Correction Prompt Construction ---
def build_correction_prompt(correction, input_text, process_context):
    return (
        f"Incorporate the following corrections to the given Input. Take into account the context if not blank and needed\n"
        f"Corrections: {correction}\nInput: {input_text}"
        f"Context: {process_context}\n"
    )

# Stage 1 Functions
def submit_correction_1(correction, output1, chat):
    # Only last correction is relevant
    if correction.strip():
        chat = [(correction, "")]
    else:
        chat = []
    return chat, gr.update(value="")  # Clear correction box

def generate_1(prompt, chat, stage_outputs, planet_name, output1, process_context):
    # If chat has a correction, use correction prompt, else use normal template
    if chat and chat[-1][0].strip():
        correction = chat[-1][0]
        input_text = output1
        formatted_prompt = build_correction_prompt(correction, input_text)
        template = None  # Not used in correction mode
        corrections = correction
    else:
        context = f"Initial Prompt: {prompt}\n\n\nProcess Context: {process_context}"
        corrections = ""
        template = prompt_templates[1]
        formatted_prompt = template.format(context=context, corrections=corrections)
    deployment_name = planet_to_deployment[planet_name]
    llm_generator = get_llm_response(formatted_prompt, corrections, template, deployment_name)
    last_chunk = ""
    for chunk in llm_generator:
        last_chunk = chunk
        yield chunk
    # After generation, autosave
    stage_outputs[1] = last_chunk

def save_1(output_text, stage_outputs):
    stage_outputs[1] = output_text
    return stage_outputs

def save_context(context_value, stage_outputs):
    # Example: store context in stage_outputs or another state
    stage_outputs["context"] = context_value
    return stage_outputs

# Stage 2-4 Functions
def submit_correction_n(correction, output_box, chat):
    if correction.strip():
        chat = [(correction, "")]
    else:
        chat = []
    return chat, gr.update(value="")  # Clear correction box

def generate_n(stage_num, chat, stage_outputs, planet_name, output_box, process_context):
    if chat and chat[-1][0].strip():
        correction = chat[-1][0]
        input_text = output_box
        formatted_prompt = build_correction_prompt(correction, input_text, process_context)
        template = None
        corrections = correction
    else:
        context = get_stage_context(stage_num, stage_outputs) + f"\n\n\nProcess Context: {process_context}"
        corrections = ""
        template = prompt_templates[stage_num]
        formatted_prompt = template.format(context=context, corrections=corrections)
    deployment_name = planet_to_deployment[planet_name]
    llm_generator = get_llm_response(formatted_prompt, corrections, template, deployment_name)
    last_chunk = ""
    for chunk in llm_generator:
        last_chunk = chunk
        yield chunk
    # After generation, autosave
    stage_outputs[stage_num] = last_chunk

def save_n(stage_num, output_text, stage_outputs):
    stage_outputs[stage_num] = output_text
    return stage_outputs

# Gradio UI Setup
with gr.Blocks(title="AIPROCS", theme=gr.themes.Soft()) as app:
    stage_outputs = gr.State({1: "", 2: "", 3: "", 4: ""})

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                list(planet_to_deployment.keys()),
                value=list(planet_to_deployment.keys())[0],
                label="Select Model",
                interactive=True
            )
        with gr.Column(scale=9):
            pass  # Placeholder for main content

    with gr.Tabs() as stages:
        # Stage 1
        with gr.Tab("Overview"):
            with gr.Row():
                gen_btn1 = gr.Button("Generate", scale=0)
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Initial Prompt", lines=3)
                    correction1 = gr.Textbox(label="New Correction")
                    chat1 = gr.Chatbot(label="Corrections History", height=height_correction)
                    process_context = gr.Textbox(label="Process Context", lines=2, value="", interactive=True)
                output1 = gr.Textbox(label="Current Output", interactive=True, lines=15)
        
        # Stage 2
        with gr.Tab("Roles & Responsibilities"):
            with gr.Row():
                gen_btn2 = gr.Button("Generate", scale=0)
            with gr.Row():
                with gr.Column():
                    chat2 = gr.Chatbot(label="Corrections History", height=height_correction)
                    correction2 = gr.Textbox(label="New Correction")
                output2 = gr.Textbox(label="Current Output", interactive=True, lines=15)
        
        # Stage 3
        with gr.Tab("Diagram"):
            with gr.Row():
                gen_btn3 = gr.Button("Generate", scale=0)
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
            with gr.Row():
                with gr.Column():
                    chat4 = gr.Chatbot(label="Corrections History", height=height_correction)
                    correction4 = gr.Textbox(label="New Correction")
                output4 = gr.Textbox(label="Current Output", interactive=True, lines=15)

    # Event Handlers
    # Stage 1: Output box auto-save on change
    output1.change(
        save_1, [output1, stage_outputs], [stage_outputs]
    )
    # Stage 1: Correction submit triggers both chat update and generation
    correction1.submit(
        submit_correction_1, [correction1, output1, chat1], [chat1, correction1]
    ).then(
        generate_1, [prompt, chat1, stage_outputs, model_selector, output1, process_context], output1
    )
    gen_btn1.click(
        generate_1, [prompt, chat1, stage_outputs, model_selector, output1, process_context], output1
    )

    process_context.change(
    save_context, [process_context, stage_outputs], [stage_outputs]
)

    # Stages 2-4: Output box auto-save on change
    for i in range(2, 5):
        output_box = globals()[f"output{i}"]
        output_box.change(
            save_n, [gr.State(i), output_box, stage_outputs], [stage_outputs]
        )
    # Stages 2-4: Correction submit triggers both chat update and generation
    for i in range(2, 5):
        correction_box = globals()[f"correction{i}"]
        chat_box = globals()[f"chat{i}"]
        gen_btn = globals()[f"gen_btn{i}"]
        output_box = globals()[f"output{i}"]

        correction_box.submit(
            submit_correction_n, [correction_box, output_box, chat_box], [chat_box, correction_box]
        ).then(
            generate_n,
            inputs=[gr.State(i), chat_box, stage_outputs, model_selector, output_box, process_context],
            outputs=output_box
        )

        gen_btn.click(
            generate_n,
            inputs=[gr.State(i), chat_box, stage_outputs, model_selector, output_box, process_context],
            outputs=output_box
        )


# FastAPI integration
fastapi_app = FastAPI()
mount_gradio_app(fastapi_app, app, path="")

if __name__ == "__main__":
    app.launch()