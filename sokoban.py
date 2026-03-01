import gradio as gr
from graph.graph import SokobanChat


async def setup():
    chat = SokobanChat()
    await chat.setup()
    return chat


async def cleanup():
    chat = SokobanChat()
    await chat.setup()
    return chat, None, None, None


async def file_setup(chat, file_path=None):
    results = await chat.file_setup(file_path)
    return chat, results


async def process_sokoban_file(chat):
    if not chat.file_path_upload:
        return chat, "Please upload the sokoban game file first!"
    results = await chat.run_superstep()
    return chat, results["content"]


with gr.Blocks(theme=gr.themes.Default(primary_hue="emerald")) as demo:
    gr.Markdown("## Sokoban Game Assistant")
    sokoban_chat = gr.State()

    with gr.Row():
        input_sokoban_game_file = gr.File(label="Upload Sokoban Game File")
        output_file = gr.Textbox(label="Sokoban Game Initial State")

    with gr.Row():
        load_button = gr.Button("Load Game File", variant="primary")
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Submit", variant="primary")

    with gr.Row():
        chatbot = gr.Textbox(label="Game Assistant AI")

    demo.load(setup, [], [sokoban_chat])

    load_button.click(fn=file_setup, inputs=[sokoban_chat, input_sokoban_game_file], outputs=[sokoban_chat, output_file])
    go_button.click(fn=process_sokoban_file, inputs=[sokoban_chat], outputs=[sokoban_chat, chatbot])
    reset_button.click(fn=cleanup, inputs=[], outputs=[sokoban_chat, chatbot, input_sokoban_game_file, output_file])

demo.launch(share=True, auth=None)
