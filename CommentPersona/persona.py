import asyncio
from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from time import time
from jupyterlab_chat.models import Message, NewMessage
from jupyter_ydoc.ynotebook import YNotebook
from jupyter_server.base.call_context import CallContext
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import ToolMessage
from typing_extensions import TypedDict
from typing import Optional
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any, Tuple
from jupyter_server_ai_tools import find_tools, run_tools
from typing import Optional, Tuple
from jupyter_ydoc.ynotebook import YNotebook
from pycrdt import Text  # or y_py.YText depending on backend
from collections import defaultdict
from .agent import run_langgraph_agent as external_run_langgraph_agent
from langchain_core.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor

from .agent import run_supervisor_agent


from langchain.prompts import PromptTemplate





#from ypy_websocket.ystore import BaseYStore
#from ypy_websocket.yutils import YMessageType
import json
import pprint
from dotenv import load_dotenv
import os
from datetime import datetime
import pytz

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

"""
------- DEMO - MULTI-AGENT -------
user: can you watch my document?

agent: Absolutely! Would you me like me to suggest edits via comments or clean things up for you as you go? 

user: I would like you to edit the doc as I go. 

agent: Great! Collaborative session starting now...


cell 2: Replaced "for i in range("five")" with "for i in range(5)". 

cell 3: Removed ""


"""


"""
TODO: 

1. add in some customization for tone 
2. add in simple supervisor 
3. write end_observation method
4. confirm multiple personas can do this at once


- Some kind of cursor update for the persona when it's writing
- be able to tell the active notebook when multiple notebooks are open
     the_room_id = "JupyterLab:globalAwareness" 

        doc = websocket_server.rooms[the_room_id]
        self.log.info(f"JUPYTER LAB GLOBAL: {doc.awareness.states}")
- Can personas call out to eachother 

DEMO: 
- 3 peronsas 
    1. prompt to go through copy edit your markdown cells 
        - have some tone variables
        - grammar & tone

    2. Linting agent
    3. Unit Test --> open a python file and start writing  
    4. add comments

    one persona that you temperarory have in place that turns on the functionality of the 3 
    or call out individually
    THE GOAL: a single supervisor that is able to call out the other personas 

    be reusable --> no vaporware

"""



def convert_mcp_to_openai(tools: list[dict]) -> list[dict]:
    """Convert a list of MCP-style tools to OpenAI-compatible function specs."""
    openai_tools = []

    for tool in tools:
        name = tool["name"]
        description = tool.get("description", "")
        input_schema = tool.get("inputSchema", {})

        openai_tools.append({
            "name": name,
            "description": description,
            "parameters": input_schema  # This is the key OpenAI expects
        })

    return openai_tools


class State(TypedDict):
    messages: list



class CodeCommenter(BasePersona):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_cell = ""
        self._startCollab = False
        self._user_prompt = ""
        self.notebooks = {}
        self._collab_task_in_progress = False
        self.global_awareness_observer = None
        self.notebook_observers = {}


    @property
    def defaults(self):
        return PersonaDefaults(
            name="CodeCommenter",
            description="A Jupyter AI Assistant who can write to notebooks to add comments",
            avatar_path="/api/ai/static/jupyternaut.svg",
            system_prompt="You are a function-calling assistant operating inside a JupyterLab environment, use your tools to operate on the notebook!"
        )
    

    async def get_active_cell(self, notebook): 
        awareness_states = notebook.awareness.states
        self.log.info(f"[DEBUG] get_active_cell CALLED. awareness_states = {dict(awareness_states)}")
        for client_id, state in awareness_states.items():
            active_cell = state.get("activeCellId")
            self.log.info(f"👤 Client {client_id} activeCellId: {active_cell}")
            self.log.info(f"ACTIVE CELL IS: {active_cell}")
            if active_cell: 
                return active_cell

        return "NO CELL ID"
    

    def extract_current_notebook_path(self, global_awareness_doc, target_username: str) -> str | None:
            for client_id, state in global_awareness_doc.awareness.states.items():
                user = state.get("user", {})
                username = user.get("username")
                if username == target_username:
                    current = state.get("current")
                    if current and current.startswith("notebook:"):
                        return current.removeprefix("notebook:RTC:")  # Python 3.9+
            return None
        
    async def get_active_notebook(self, client_id: str, notebook_path: str) -> YNotebook | None:
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        collaboration = serverapp.web_app.settings["jupyter_server_ydoc"]
        websocket_server = collaboration.ywebsocket_server

        for room_id in websocket_server.rooms:
            try:
                doc = await collaboration.get_document(room_id=room_id, copy=False)
                path = getattr(doc, "path", None)
                self.log.info(f"Path: {path}")
                if isinstance(doc, YNotebook) and getattr(doc, "path", None) == notebook_path:
                    self.log.info(f"✅ Matched path '{notebook_path}' in room {room_id}")
                    awareness_states = doc.awareness.states
                    self.log.info(f"ALL STATES: {awareness_states.items()}")
                   
                    for client_id, state in awareness_states.items():
                        active_cell = state.get("activeCellId")
                        self.log.info(f"👤 Client {client_id} activeCellId: {active_cell}")
             
                    return doc
            except Exception as e:
                self.log.warning(f"⚠️ Could not inspect room {room_id}: {e}")

      
        self.log.warning(f"❌ No active notebook found for client_id: {client_id}")
        return None

    async def run_langgraph_agent(self, notebook: YNotebook, user_prompt: str, tone_prompt):
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        extension_manager = serverapp.extension_manager
        raw_tools = find_tools(extension_manager, return_metadata_only = False)

        self.log.info("ABOUT TO RUN AGENT???")

        await external_run_langgraph_agent(extension_manager, self.log, self.ychat, raw_tools, notebook, user_prompt, tone_prompt, self.id, self.get_active_cell)
    
    # here I want you to get the current contents of the notebook, so call tool_groups.get_notebook() or something, 
    # then the prompt should be something like, Hello! Take a look at the current notebook for grammar mistakes, fix them where neccesary


    async def _run_with_flag_reset(self, ynotebook, prompt, tone_prompt):
        try:
            await self.run_langgraph_agent(ynotebook, prompt, tone_prompt)
        finally:
            self._collab_task_in_progress = False

    def start_collaborative_session(self, ynotebook: YNotebook, path: str, tone_prompt):
        """
        Observes awareness (cursor position, etc) and reacts when a user changes their selection.
        """

        def on_awareness_change(event_type, data):
            #self.log.info(f"AWARENESS CHANGED!!!!!!!!!! FOR NOTEBOOK {path}")
            
            if self._collab_task_in_progress:
                #self.log.info("🔄 Agent already running — skipping awareness change.")
                return

            for clientID, state in ynotebook.awareness.states.items():
                self.log.info(" LOOKING FOR NOTEBOOKS")
               

                current_cell = state.get("activeCellId")
                self.log.info(f"CURRENT CELL: {current_cell}")
                if current_cell == None: 
                    continue

                

                last_cell = self.notebooks[path]["activeCell"]
                if current_cell != last_cell:
                    self.notebooks[path]["activeCell"] = current_cell
                    self.log.info(f"📍 Cursor changed YText for client {clientID}")
                    self.ychat.add_message(
                        NewMessage(
                            body=f"Active Cell is now: {current_cell}",
                            sender=self.id,
                        )
                    )

                    prompt = f"The user is currently editing in cell {current_cell} so, to avoid disrupting their work, DO NOT write to, or delete that cell. You can only edit in the rest of the notebook cells"

                    # ✅ Don't cancel current job — just let it finish
                    self._collab_task_in_progress = True
                    self._running_task = asyncio.create_task(
                        self._run_with_flag_reset(ynotebook, prompt, tone_prompt)
                    )

        awareness = ynotebook.awareness
        unsubscribe = awareness.observe(on_awareness_change)
        self.notebook_observers[path] = (awareness, unsubscribe)
        self.log.info(f"✅ Awareness observer registered for notebook: {path}")


    async def _handle_global_awareness_change(self, client_id, tone_prompt):
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        collaboration = serverapp.web_app.settings["jupyter_server_ydoc"]
        websocket_server = collaboration.ywebsocket_server

        the_room_id = "JupyterLab:globalAwareness"
        global_doc = websocket_server.rooms[the_room_id]
        self.log.info(f"JUPYTER LAB GLOBAL: {global_doc.awareness.states}")

        active_notebook_path = self.extract_current_notebook_path(global_doc, client_id)
        if not active_notebook_path:
            self.log.warning("❌ No active notebook path found.")
            return
        
        self.log.info(f"NBS: {self.notebooks}")
        self.log.info(f"CURRENT NB: {active_notebook_path}")
        if active_notebook_path not in self.notebooks: 
            notebook = await self.get_active_notebook(client_id, active_notebook_path)
            active_cell = await self.get_active_cell(notebook)
            self.notebooks[active_notebook_path] = {
                "activeCell": active_cell
            }
            if notebook:
                self.start_collaborative_session(notebook, active_notebook_path, tone_prompt)
            else: 
                self.log.info(f"THERE WAS NO COLLABORATIVE NOTEBOOK OBSERVER STARTED FOR {active_notebook_path}")



    async def start_global_observation(self, client_id, tone_prompt):
        """
        Observes awareness changes in global awarness
        """
        self.log.info("INSIDE STARTING GLOBAL OBS")
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        collaboration = serverapp.web_app.settings["jupyter_server_ydoc"]
        websocket_server = collaboration.ywebsocket_server

        the_room_id = "JupyterLab:globalAwareness" 

        doc = websocket_server.rooms[the_room_id]

        def on_awareness_change(event_type, data):
            asyncio.create_task(self._handle_global_awareness_change(client_id, tone_prompt))
                

        awareness = doc.awareness
        unsubscribe = awareness.observe(on_awareness_change)
        self.global_awareness_observer = (awareness, unsubscribe)
        self.log.info("✅ GLOBAL Awareness observer registered.")




    async def process_message(self, message: Message):
        client_id = message.sender

        @tool
        async def start_collaborative_session() -> str:
            """Starts a comment adding collaborative session. Optionally accepts a tone prompt."""
            await self.start_global_observation(client_id, "")
            return f"Collaborative session started."

        @tool
        async def stop_collaborative_session() -> str:
            """Stops the current collaborative comment adding session."""

            # Unregister global awareness observer
            if self.global_awareness_observer:
                awareness, unsubscribe = self.global_awareness_observer
                awareness.unobserve(unsubscribe)
                self.global_awareness_observer = None
                self.log.info("🛑 Global awareness observer removed.")

            # Unregister all notebook-level observers
            for path, (awareness, unsubscribe) in self.notebook_observers.items():
                awareness.unobserve(unsubscribe)
                self.log.info(f"🛑 Notebook awareness observer removed for: {path}")
            self.notebook_observers.clear()

            # reset per-notebook state
            self.notebooks.clear()

            return "Collaborative session stopped and all observers removed."

        async def stream_typing(full_text: str) -> str:
            """Streams a chat message to the user as if it's being typed."""
            stream_msg_id = self.ychat.add_message(NewMessage(body="", sender=self.id))
            current_text = ""

            for char in full_text:
                await asyncio.sleep(0.02)
                current_text += char
                self.ychat.update_message(
                    Message(
                        id=stream_msg_id,
                        body=current_text,
                        time=time(),
                        sender=self.id,
                        raw_time=False,
                    ),
                    append=False,
                )
            return "Typing stream completed."

        tools = [start_collaborative_session, stop_collaborative_session]
        await run_supervisor_agent(
            logger=self.log,
            stream=stream_typing,
            user_message=message.body,
            tools=tools
        )
        

        

        # Optionally stream response to chat
        #self.ychat.add_message(NewMessage(body=response, sender=self.id))

        #await start_collaborative_session(tone_prompt="Sound like a sassy russian")

        # make a simple agent with two tools


## teh fcntion ebewlosd is goin gto add two nmbers togere

                
