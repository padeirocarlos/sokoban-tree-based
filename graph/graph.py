import os
import uuid
import logging
from graph.states import SokobanState, initiate_state
from langgraph.graph import StateGraph, END
from sokoban.sokoban_tools import SokobanRules
from langgraph.checkpoint.memory import InMemorySaver
from agent.agent import convert_current_state_to_map
from edges.edges import route_after_executor_node
from nodes.nodes import move_node, executor_node, result_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sokoban-Agentic-Workflow (SAW)")

DEFAULT_TEST_FILE = os.path.join(os.getcwd(), "dataset/test/1_3.txt")
FALLBACK_TEST_FILE = os.path.join(os.getcwd(), "dataset/test/1_4.txt")


async def workflow_app(memory=None) -> StateGraph:
    """Build and compile the LangGraph workflow for Sokoban solving."""
    if memory is None:
        memory = InMemorySaver()

    workflow = StateGraph(SokobanState)
    workflow.add_node("moves", move_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("result", result_node)

    workflow.set_entry_point("moves")
    workflow.add_edge("moves", "executor")
    workflow.add_conditional_edges(
        "executor",
        route_after_executor_node,
        {"moves": "moves", "result": "result"},
    )
    workflow.add_edge("result", END)

    graph = workflow.compile(checkpointer=memory)
    try:
        graph.get_graph().draw_mermaid_png(output_file_path="dev/flowchart.png")
    except Exception as e:
        logger.warning(f"Could not generate flowchart: {e}")

    return graph


class SokobanChat:
    def __init__(self):
        self.graph = None
        self.memory = InMemorySaver()
        self.chat_id = str(uuid.uuid4())
        self.file_path_upload = False
        self.uploaded_file_path = None

    async def setup(self):
        self.graph = await workflow_app(self.memory)

    async def file_setup(self, file_path=None):
        if file_path is None:
            file_path_str = DEFAULT_TEST_FILE
            self.file_path_upload = False
        else:
            file_path_str = file_path.name
            self.file_path_upload = True
        self.uploaded_file_path = file_path_str
        sokoban_rules = SokobanRules(file_path_str)
        return convert_current_state_to_map(sokoban_rules)

    async def run_superstep(self):
        config = {"configurable": {"thread_id": self.chat_id}}
        test_file = self.uploaded_file_path or FALLBACK_TEST_FILE
        state = initiate_state(model_name="qwen3:latest", test_file=test_file)
        result = await self.graph.ainvoke(state, config=config)

        visited_map_state = "\n ------ \n".join(result['visited_map_state'])
        if result['status'] == "success":
            message = "Congratulation! You solved the puzzle!"
        else:
            message = "The AI fails to solve it. Try it later!"

        final_response = (
            f" Puzzle State: {visited_map_state}\n ------ \n"
            f" | {message}\n Puzzle Move: {result['moves']}\n"
        )
        return {"role": "assistant", "content": final_response}