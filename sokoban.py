import uuid
import logging
from pathlib import Path
from graph.graph import workflow_app
from sokoban.sokoban_tools import SokobanRules
from graph.states import initiate_state

logger = logging.getLogger("Sokoban-Agentic-Workflow (SAW)")

async def main():
    
    parent_dir =  Path.cwd()
    
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    
    data_file = parent_dir / "dataset/test/4_1.txt" 

    initial_sokoban = SokobanRules(data_file)
    
    logger.info(str(initial_sokoban.serialize_map()))
    
    agent_state = initiate_state(initial_sokoban, "ollama3")
    
    app = workflow_app()
    
    result = await app.ainvoke(agent_state, config=config)
    
    final_response = result["final_response"]
    

if __name__ == "__main__":
    main()