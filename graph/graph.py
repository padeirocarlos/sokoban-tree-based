import os
import uuid
import logging
from graph.states import SokobanState
from graph.states import initiate_state
from langgraph.graph import StateGraph, END
from sokoban.sokoban_tools import SokobanRules
from langgraph.checkpoint.memory import InMemorySaver
from agent.agent import convert_current_state_to_map

from edges.edges import (
    route_after_executor_node,
)
from nodes.nodes import (
    move_node,
    executor_node,
    result_node,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sokoban-Agentic-Workflow (SAW)")

async def workflow_app(sql_memory = InMemorySaver()) -> StateGraph:
    
    sql_memory = sql_memory
    workflow = StateGraph(SokobanState)
    
    # Add nodes
    workflow.add_node("moves", move_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("result", result_node)

    # set entry point to decomposition node
    workflow.set_entry_point("moves")
    workflow.add_edge("moves", "executor")
    
    workflow.add_conditional_edges("executor", route_after_executor_node,
        {
            "moves": "moves",
            "result": "result",
        }
    )
    
    workflow.add_edge("result", END)
    graph = workflow.compile(checkpointer=sql_memory)
    graph.get_graph().draw_mermaid_png(output_file_path="dev/flowchart.png")
    
    return graph

class SokobanChat:
    def __init__(self):
        self.graph = None
        self.interaction_number = 0
        self.memory = InMemorySaver()
        self.sokobanChat_id = str(uuid.uuid4())
        self.file_path_upload = False
        self.uploaded_file_path = None

    async def setup(self):
        await self.build_graph()

    async def file_setup(self, file_path=None):
        if file_path is None:
            file_path_str = os.path.join(os.getcwd(), "dataset/test/1_3.txt")
            self.file_path_upload = False
        else:
            file_path_str = file_path.name
            self.file_path_upload = True
        self.uploaded_file_path = file_path_str
        sokoban_rules = SokobanRules(file_path_str)
        return convert_current_state_to_map(sokoban_rules)

    async def build_graph(self):
        self.graph = await workflow_app(self.memory)

    async def run_superstep(self):
        config = {"configurable": {"thread_id": self.sokobanChat_id}}

        test_file = self.uploaded_file_path or os.path.join(os.getcwd(), "dataset/test/1_4.txt")
        state = initiate_state(model_name="qwen3:latest", test_file=test_file)
        self.interaction_number = state["max_iterations"]
        result = await self.graph.ainvoke(state, config=config)
        
        # This for nake sure that first response should be from final_response 
        visited_map_state = "\n ------ \n".join(result['visited_map_state'])
        if str(result['status']).lower() == ("success").lower():
            final_response = f" Puzzle State: {visited_map_state}  \n ------ \n | 🔬 🚀 Congratulation 🚀 You solved the puzzle!  \n Puzzle Move: {result['moves']} \n"      
        else:
            final_response = f" Puzzle State: {visited_map_state}  \n ------ \n | 🔬 ⚠️ The AI fails to solve it. Try it later 🏄🏽!  \n Puzzle Move: {result['moves']} \n"
        
        return {"role": "assistant", "content": final_response}