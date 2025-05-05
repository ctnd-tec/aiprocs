import gradio as gr

def generate_response(stage_num, user_input, history, prev_stages):
    # placeholder for actual LLM calls
    context = "\n".join([f"Stage {i+1}: {output}" for i, output in enumerate(prev_stages[:stage_num-1])])
    response = f"Stage {stage_num} response to: {user_input}\nContext:\n{context}"
    return response

def handle_stage1(user_input, chat_history, stage1_out, stage2_out, stage3_out):
    prev_stages = [stage1_out, stage2_out, stage3_out]
    response = generate_response(1, user_input, chat_history, prev_stages)
    chat_history.append((user_input, response))
    return chat_history, response

def handle_stage2(user_input, chat_history, stage1_out, stage2_out, stage3_out):
    prev_stages = [stage1_out, stage2_out, stage3_out]
    response = generate_response(2, user_input, chat_history, prev_stages)
    chat_history.append((user_input, response))
    return chat_history, response

def handle_stage3(user_input, chat_history, stage1_out, stage2_out, stage3_out):
    prev_stages = [stage1_out, stage2_out, stage3_out]
    response = generate_response(3, user_input, chat_history, prev_stages)
    chat_history.append((user_input, response))
    return chat_history, response

with gr.Blocks() as app:
    stage1_out = gr.State("")
    stage2_out = gr.State("")
    stage3_out = gr.State("")
    
    with gr.Tabs() as tabs:
        with gr.TabItem("Stage 1") as tab1:
            stage1_chat = gr.Chatbot()
            stage1_input = gr.Textbox(label="Your message")
            stage1_submit = gr.Button("Submit")
            stage1_next = gr.Button("Next")

        with gr.TabItem("Stage 2") as tab2:
            stage2_chat = gr.Chatbot()
            stage2_input = gr.Textbox(label="Your message")
            stage2_submit = gr.Button("Submit")
            stage2_next = gr.Button("Next")

        with gr.TabItem("Stage 3") as tab3:
            stage3_chat = gr.Chatbot()
            stage3_input = gr.Textbox(label="Your message")
            stage3_submit = gr.Button("Submit")

    stage1_submit.click(
        handle_stage1,
        [stage1_input, stage1_chat, stage1_out, stage2_out, stage3_out],
        [stage1_chat, stage1_out]
    )
    
    stage1_next.click(
        lambda: gr.Tabs(selected=1),
        outputs=[tabs]
    )

    stage2_submit.click(
        handle_stage2,
        [stage2_input, stage2_chat, stage1_out, stage2_out, stage3_out],
        [stage2_chat, stage2_out]
    )
    
    stage2_next.click(
        lambda: gr.Tabs(selected=2),
        outputs=[tabs]
    )

    stage3_submit.click(
        handle_stage3,
        [stage3_input, stage3_chat, stage1_out, stage2_out, stage3_out],
        [stage3_chat, stage3_out]
    )

app.launch()