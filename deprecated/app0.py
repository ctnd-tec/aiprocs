import gradio as gr
from typing import Dict, Any

class AppState:
    def __init__(self):
        self.steps = {
            "overview": {"data": None, "valid": False},
            "roles": {"data": None, "valid": False},
            "diagram": {"data": None, "valid": False},
            "document": {"data": None, "valid": False}
        }
        self.current_step = 0
        self.tab_names = ["overview", "roles", "diagram", "document"]  # Add this line

    def mark_stale(self, from_step):
        start_idx = self.tab_names.index(from_step) + 1
        for step in self.tab_names[start_idx:]:
            self.steps[step]["valid"] = False

# Custom CSS to hide native tab navigation
css = """
.tab-nav { display: none !important; }
.stale-tab { border: 2px solid #ff4444 !important; }
"""

def update_ui(state):
    #tab_update = gr.Tabs.update(selected=state.value.current_step)
    indicators = {
        name: gr.Button.update(variant="secondary" if state.value.steps[name]["valid"] else "primary")
        for name in state.tab_names
    }
    #return tab_update, *indicators.values()
    return tuple(indicators.values())



with gr.Blocks(css=css) as app:
    state = gr.State(AppState())
    
    # Navigation controls
    with gr.Row():
        back_btn = gr.Button("Back", visible=False)
        next_btn = gr.Button("Next", variant="primary")
        overview_ind = gr.Button("1", min_width=30)
        roles_ind = gr.Button("2", min_width=30)
        diagram_ind = gr.Button("3", min_width=30)
        doc_ind = gr.Button("4", min_width=30)
    
    # Main content tabs
    with gr.Tabs(elem_classes="tab-nav") as tabs:
        # Overview Tab
        with gr.Tab("Overview", id="overview") as tab1:
            inp_process = gr.Textbox(label="Process Description")
            out_paraphrased = gr.Textbox(label="Paraphrased Process")
            btn_gen_overview = gr.Button("Generate")
        
        # Roles Tab
        with gr.Tab("Roles", id="roles") as tab2:
            out_roles = gr.Textbox(label="Roles & Responsibilities")
            btn_gen_roles = gr.Button("Regenerate")
        
        # Diagram Tab
        with gr.Tab("Diagram", id="diagram") as tab3:
            out_diagram = gr.Markdown("Mermaid diagram here")
            btn_gen_diagram = gr.Button("Regenerate")
        
        # Document Tab
        with gr.Tab("Document", id="document") as tab4:
            out_doc = gr.Textbox(label="Final Document")
            btn_gen_doc = gr.Button("Regenerate")

    # Event handlers
    def handle_navigation(change):
        def wrapper(state):
            state.current_step = max(0, min(3, state.current_step + change))
            return update_ui(state)
        return wrapper

    # back_btn.click(
    #     fn=lambda s: handle_navigation(-1)(s),
    #     inputs=state,
    #     outputs=[tabs, overview_ind, roles_ind, diagram_ind, doc_ind]
    # )

    # next_btn.click(
    #     fn=lambda s: handle_navigation(1)(s),
    #     inputs=state,
    #     outputs=[tabs, overview_ind, roles_ind, diagram_ind, doc_ind]
    # )

    back_btn.click(
    fn=lambda s: setattr(s, "current_step", max(0, s.current_step - 1)) or update_ui(s),
    inputs=state,
    outputs=[overview_ind, roles_ind, diagram_ind, doc_ind]
    )

    next_btn.click(
        fn=lambda s: setattr(s, "current_step", min(3, s.current_step + 1)) or update_ui(s),
        inputs=state,
        outputs=[overview_ind, roles_ind, diagram_ind, doc_ind]
    )

    def handle_generation(step):
        def wrapper(inputs, state):
            # Your LLM generation logic here
            state.steps[step]["data"] = f"Generated {step}"
            state.steps[step]["valid"] = True
            state.mark_stale(step)
            return update_ui(state)
        return wrapper

    btn_gen_overview.click(
        fn=handle_generation("overview"),
        inputs=[inp_process, state],
        outputs=[state, tabs, overview_ind, roles_ind, diagram_ind, doc_ind]
    )

    # Similar handlers for other generation buttons
    # ...

    # Staleness indicators
    indicators = [overview_ind, roles_ind, diagram_ind, doc_ind]
    for btn, step in zip(indicators, state.value.tab_names):
        btn.click(
            # fn=lambda s, st=step: (s.update(current_step=st), update_ui(s))[1],
            fn=lambda s, st=step: (setattr(s, "current_step", s.tab_names.index(st)), update_ui(s))[1],
            inputs=state,
            outputs=[tabs] + indicators
        )

if __name__ == "__main__":
    app.launch()