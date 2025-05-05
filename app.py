import gradio as gr
import os 

height_correction = 200

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

# def simulate_llm(context, corrections, template=None):
#     """Mock LLM function for demonstration purposes"""
#     return f"""Prompt Template:
# {template}

# Context:
# {context}

# Corrections:
# {corrections}

# Generated Text: [LLM output based on above]
# """

# succinct version of simulate_llm
def simulate_llm(context, corrections, template=None):
    """Mock LLM function for demonstration purposes"""
    return f"Context:\n{context}\n\nCorrections:\n{corrections}\n\nGenerated Text: [LLM output based on above]"

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
    return simulate_llm(formatted_prompt, corrections, template)


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
    return simulate_llm(formatted_prompt, corrections, template)

def save_n(stage_num, output_text, stage_outputs):
    stage_outputs[stage_num] = output_text
    return stage_outputs

with gr.Blocks(title="LLM Wizard",theme=gr.themes.Soft()) as app:
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
                    # gen_btn1 = gr.Button("Generate")
                output1 = gr.Textbox(label="Current Output", interactive=True, lines=15)
            # save_btn1 = gr.Button("Save Edits & Lock Stage")
        
        # Stage 2
        with gr.Tab("Roles & Responsibilities"):
            with gr.Row():
                gen_btn2 = gr.Button("Generate", scale=0)
                save_btn2 = gr.Button("Save", scale=0)
            with gr.Row():
                with gr.Column():
                    chat2 = gr.Chatbot(label="Corrections History", height=height_correction)
                    correction2 = gr.Textbox(label="New Correction")
                    #gen_btn2 = gr.Button("Generate")
                output2 = gr.Textbox(label="Current Output", interactive=True, lines=15)
            #save_btn2 = gr.Button("Save Edits & Lock Stage")
        
        # Stage 3
        with gr.Tab("Diagram"):
            with gr.Row():
                gen_btn3 = gr.Button("Generate", scale=0)
                save_btn3 = gr.Button("Save", scale=0)
            with gr.Row():
                with gr.Column():
                    chat3 = gr.Chatbot(label="Corrections History", height=height_correction)
                    correction3 = gr.Textbox(label="New Correction")
                    #gen_btn3 = gr.Button("Generate")
                output3 = gr.Textbox(label="Current Output", interactive=True, lines=15)
                mermaid_diagram = gr.Markdown(label="Mermaid Diagram",  value="""
                    ```mermaid
                    graph LR
                        A --> B 
                        B --> C
                    ```
                    """)
                # rerender_btn = gr.Button("Re-render Diagram")  # button to re-render the diagram until the fix is implemented
                # see https://github.com/gradio-app/gradio/issues/11073
            #save_btn3 = gr.Button("Save Edits & Lock Stage")
        
        # Stage 4
        with gr.Tab("Full text"):
            with gr.Row():
                gen_btn4 = gr.Button("Generate", scale=0)
                save_btn4 = gr.Button("Save", scale=0)
            with gr.Row():
                with gr.Column():
                    chat4 = gr.Chatbot(label="Corrections History", height=height_correction)
                    correction4 = gr.Textbox(label="New Correction")
                    #gen_btn4 = gr.Button("Generate")
                output4 = gr.Textbox(label="Current Output", interactive=True, lines=15)
            #save_btn4 = gr.Button("Save Edits & Lock Stage")


# EVENTS


    # Stage 1 Events
    correction1.submit(submit_correction_1, [correction1, chat1], [chat1]).then(
        lambda: "", None, correction1)
    gen_btn1.click(generate_1, [prompt, chat1, stage_outputs], output1)
    save_btn1.click(save_1, [output1, stage_outputs], [stage_outputs])

    # Stage 2-4 Events
    for i in range(2, 5):
        correction_box = locals()[f"correction{i}"]
        chat_box = locals()[f"chat{i}"]
        gen_btn = locals()[f"gen_btn{i}"]
        output_box = locals()[f"output{i}"]
        save_btn = locals()[f"save_btn{i}"]
        
        correction_box.submit(
            submit_correction_n, [correction_box, chat_box], [chat_box]
        ).then(lambda: "", None, correction_box)
        
        gen_btn.click(
            lambda stage_num, chat, outputs: generate_n(stage_num, chat, outputs),
            inputs=[gr.State(i), chat_box, stage_outputs],
            outputs=output_box
        )
        
        save_btn.click(
            lambda stage_num, out, outputs: save_n(stage_num, out, outputs),
            inputs=[gr.State(i), output_box, stage_outputs],
            outputs=stage_outputs
        )


    # def rerender_mermaid():
    #     return """
    #         ```mermaid
    #         graph LR
    #             A --> B
    #             B --> C   
    #         ```
    
    #         """

    # rerender_btn.click(rerender_mermaid, None, mermaid_diagram)

if __name__ == "__main__":
    app.launch()