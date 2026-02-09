import logging
import gradio as gr
from graph.graph import SokobanChat

logger = logging.getLogger("Sokoban-Agentic-Workflow (SAW)")

# Setup all setting of the Sokoban Game 
async def setup():
    sokobanChat = SokobanChat()
    await sokobanChat.setup()
    return sokobanChat

# Reset all setting of the Sokoban Game 
async def cleanup():
    new_sokobanChat = SokobanChat()
    await new_sokobanChat.setup()
    return  new_sokobanChat, None, None, None

# Upload the file in the Sokoban Game 
async def file_setup(sokobanChat, file_path = None):
    results = await sokobanChat.file_setup(file_path)
    return sokobanChat, results

# Process the sokoban game file using AI Agentic and Agent Model
async def process_sokoban_file(sokobanChat):
    if not sokobanChat.file_path_upload:
        return sokobanChat,  "#### ⚠️ Please upload the sokoban game file to Sokoban Assistant (SSA) AI first!"
    results = await sokobanChat.run_superstep()
    return sokobanChat, results["content"]

def free_resources(sokobanChat):
    try:
        if sokobanChat:
            sokobanChat.setup()
    except Exception as e:
        print(f"Exception during cleanup: {e}")
        
with gr.Blocks(theme=gr.themes.Default(primary_hue="emerald")) as demo:
    gr.Markdown("## Sokoban Game Assistant Supporter ")
    sokobanChat = gr.State(delete_callback=free_resources)
    
    with gr.Row():
        # input_sokoban_game_file = gr.UploadButton(label = "Upload Sokoban Game File...!")
        input_sokoban_game_file = gr.File(label = "Upload Sokoban Game File...!")
        output_file = gr.Textbox(label="ℹ️ Sokoban Game Initial State!")
    
    with gr.Row():
        sokoban_game_file_button = gr.Button("Sokoban Game File Saving...!", variant="primary")
        reset_button = gr.Button("❌ Reset", variant="stop")
        go_button = gr.Button("🚀 Submit", variant="primary")
           
    with gr.Row():
        chatbot = gr.Textbox(label= "📝 Game Assistant AI ✅ ") 

    demo.load(setup, [], [sokobanChat])
    
    sokoban_game_file_button.click(fn=file_setup, inputs=[sokobanChat,input_sokoban_game_file], outputs=[sokobanChat, output_file])
    go_button.click(fn=process_sokoban_file, inputs=[sokobanChat], outputs=[sokobanChat, chatbot])
    reset_button.click(fn=cleanup, inputs=[], outputs=[sokobanChat, chatbot, input_sokoban_game_file, output_file])
    
demo.launch(share=True, auth=None)
