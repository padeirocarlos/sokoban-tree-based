import os
import time
import uuid
import logging
from pathlib import Path
from graph.states import SokobanState
from graph.states import initiate_state
from langgraph.graph import StateGraph, END
from sokoban.sokoban_tools import SokobanRules
from langgraph.checkpoint.memory import InMemorySaver
from sokoban.sokoban_tools import global_sokobanGame

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
            "END": END,
            "moves": "moves",
            "result": "result",
        }
    )
    
    workflow.add_edge("result", END)
    graph = workflow.compile(checkpointer=sql_memory)
    graph.get_graph().draw_mermaid_png(output_file_path="dev/sokoban.png")
    
    return graph

class SokobanChat:
    def __init__(self):
        self.graph = None
        self.interaction_number = 0
        self.memory = InMemorySaver()
        self.sokobanChat_id = str(uuid.uuid4())
        self.initial_sokoban = None

    async def setup(self):
        await self.build_graph()
        self.initial_sokoban = SokobanRules(os.path.join(os.getcwd(), "dataset/test/4_1.txt"))
    
    async def build_graph(self):
        # Set up Graph Builder with State
        self.graph = await workflow_app(self.memory)
     
    async def run_superstep(self):
        """
        :param self: Description
        :param message: Description
        :param history: Description
        """
        
        config = {"configurable": {"thread_id": self.sokobanChat_id}}
        
        # initial_sokoban = SokobanRules(os.path.join(os.getcwd(), "dataset/test/4_1.txt"))
        # initial_sokoban = SokobanRules(os.path.join(os.getcwd(), "dataset/test/1_0.txt"))
        
        logger.info(f"\n {str(self.initial_sokoban.serialize_map())}")
        
        # gpt-oss:20b llama3:latest mistral:latest ollama3 qwen3 ayansh03/agribot
        state = initiate_state(self.initial_sokoban, "qwen3") 
        
        self.interaction_number = state["max_iterations"]
        
        result = await self.graph.ainvoke(state, config=config)
        
        # This for nake sure that first response should be from final_response 
        final_response = result["final_response"]
        
        if not final_response:
            final_response = result.get("answers", None)
        
        return {"role": "assistant", "content": final_response}