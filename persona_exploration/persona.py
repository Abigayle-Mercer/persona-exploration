



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
from langchain_core.messages import ToolMessage
from typing import Dict, Any, Tuple
from jupyter_server_ai_tools import find_tools, run_tools
from typing import Optional, Tuple
from jupyter_ydoc.ynotebook import YNotebook
from pycrdt import Text  # or y_py.YText depending on backend
from collections import defaultdict





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





class State(TypedDict):
    messages: list

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



class GrammarEditor(BasePersona):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_cell = ""
        self._startCollab = False
        self._user_prompt = ""
        self.notebooks = {}

    @property
    def defaults(self):
        return PersonaDefaults(
            name="GrammarEditor",
            description="A Jupyter AI Assistant who can write to notebooks to fix grammar mistakes",
            avatar_path="/api/ai/static/jupyternaut.svg",
            system_prompt="You are a function-calling assistant operating inside a JupyterLab environment, use your tools to operate on the notebook!"
        )
    

    async def get_active_cell(self, notebook): 
        awareness_states = notebook.awareness.states
        self.log.info(f"ALL STATES: {awareness_states.items()}")
        for client_id, state in awareness_states.items():
            active_cell = state.get("activeCellId")
            self.log.info(f"👤 Client {client_id} activeCellId: {active_cell}")
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


    async def run_langgraph_agent(self, notebook: YNotebook, user_prompt: str):
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        extension_manager = serverapp.extension_manager
        raw_tools = find_tools(extension_manager, return_metadata_only = False)
        self.log.info(f"TOOL GROUPS: {raw_tools}")
        tool_groups = {t["metadata"]["name"]: t for t in raw_tools}
        tools = [t["metadata"] for t in raw_tools]


        self.log.info(f"TOOLS: {tools}")

        memory = MemorySaver()
        llm = ChatOpenAI(api_key=api_key, model="gpt-4", temperature=0, streaming=True)
        openai_functions = convert_mcp_to_openai(tools)
        model = llm.bind_tools(tools=openai_functions)

        def parse_openai_tool_call(call: Dict) -> Tuple[str, Dict]:
            """
            Parses an OpenAI-style function call object and injects live objects like
            `ynotebook` or `scheduler` into the tool arguments based on the tool name.

            Returns:
                A tuple of (tool_name, arguments_dict)
            """
            fn = call.get("function", {})
            name = fn.get("name")
            arguments = fn.get("arguments", "{}")

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            self.log.info(f"TOOL NAME: {name}")

            # Inject the notebook for non-git/scheduler tools
            if not name.startswith("git_"):
                self.log.info("📎 Injecting ynotebook into tool arguments")
                arguments["ynotebook"] = notebook

            return name, arguments
        
        async def agent(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state["messages"]
            stream_message_id = None
            full_chunk: Optional[AIMessageChunk] = None

            async for chunk in model.astream(messages):
                if full_chunk is None:
                    full_chunk = chunk
                else:
                    full_chunk += chunk  # ✅ merges tool_call_chunks safely

                content = getattr(chunk, "content", "")
                if content:
                    if stream_message_id is None:
                        stream_message_id = self.ychat.add_message(NewMessage(body=content, sender=self.id))
                    else:
                        self.ychat.update_message(
                            Message(
                                id=stream_message_id,
                                body=content,
                                time=time(),
                                sender=self.id,
                                raw_time=False,
                            ),
                            append=True,
                        )

            # Final full message: tool_calls are now correctly parsed
            full_message = AIMessage(
                content=full_chunk.content if full_chunk else "",
                additional_kwargs=full_chunk.additional_kwargs if full_chunk else {},
                response_metadata=full_chunk.response_metadata if full_chunk else {}
            )


            self.log.info("✅ full_message: %s", full_message)
            return {"messages": messages + [full_message]}
        def should_continue(state):
            last_message = state["messages"][-1]
            return "continue" if "tool_calls" in last_message.additional_kwargs else "end"


        async def call_tool(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state["messages"]
            last_msg = messages[-1]
            tool_calls = last_msg.additional_kwargs.get("tool_calls", [])

            results = []

            for call in tool_calls:
                tool_name = call["function"]["name"]

                # write calling tool to a new text file like tool_calls.txt instead of writng calling tool to the ychat
            

                calling_msg = f"🔧 Calling {tool_name}...\n" 
                stream_msg_id = self.ychat.add_message(NewMessage(
                    body="",
                    sender=self.id,
                ))
                for char in calling_msg:
                    await asyncio.sleep(0.01)  # small delay to simulate streaming
                    self.ychat.update_message(
                        Message(
                            id=stream_msg_id,
                            body=char,
                            time=time(),
                            sender=self.id,
                            raw_time=False,
                        ),
                        append=True,
                    )

                result = await run_tools(
                    extension_manager,
                    [call],
                    parse_fn=parse_openai_tool_call,  # this will work bc I have a lookup table in the run function
                )
                
                results.append((call, result[0]))

            tool_messages = [
                ToolMessage(
                    name=call["function"]['name'],
                    content=str(result),
                    tool_call_id=call['id']
                )
                for call, result in results
            ]

            return {"messages": state["messages"] + tool_messages}

        workflow = StateGraph(State)
        workflow.add_node("agent", agent)
        workflow.add_edge(START, "agent")
        workflow.add_node("call_tool", call_tool)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"continue": "call_tool", "end": END})
        workflow.add_edge("call_tool", "agent")

        compiled = workflow.compile(checkpointer=memory)

        nb = tool_groups["read_notebook"]["callable"](notebook)

        system_prompt = f"""
        You are a function-calling assistant operating inside a JupyterLab environment.
        Your job is to read the entire notebook and if you find grammar mistakes in markdown cells, 
        doc strings, or comments, fix them using your available tools. 
        Please start by calling read_notebook
        """
        self.log.info(f"PROMPT: {system_prompt}")
        state = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        } 
        config = {"configurable": {"thread_id": "thread-1"}}

        await compiled.ainvoke(state, config=config)


        # here I want you to get the current contents of the notebook, so call tool_groups.get_notebook() or something, 
        # then the prompt should be something like, Hello! Take a look at the current notebook for grammar mistakes, fix them where neccesary


    def start_collaborative_session(self, ynotebook: YNotebook):
        """
        Observes awareness (cursor position, etc) and reacts when a user changes their selection.
        """

        def on_awareness_change(event_type, data):
            for clientID, state in ynotebook.awareness.states.items():

                cursors = state.get("cursors", [])
                if not cursors:
                    continue

                current_cell = state.get("activeCellId")

                last_cell = self._last_cell
                if current_cell != last_cell:
                    self._last_cell = current_cell

                    # Cursor moved to a different YText — likely a different cell
                    self.log.info(f"📍 Cursor changed YText for client {clientID}")
                    self.ychat.add_message(
                        NewMessage(
                            body=f"Active Cell is now: {current_cell}",
                            sender=self.id,
                        )
                    )
                    if self._startCollab:
                        prompt = f"The user is currently editing in cell {current_cell} so, to avoid disrupting their work, DO NOT write to, or delete that cell. You can only edit in the rest of the notebook cells"
                        

                        self._running_task = getattr(self, "_running_task", None)
                        if self._running_task and not self._running_task.done():
                            self._running_task.cancel()

                        self._running_task = asyncio.create_task(self.run_langgraph_agent(ynotebook, prompt))
                        # call the agent here? with the user prompt somehow? 

        awareness = ynotebook.awareness
        awareness.observe(on_awareness_change)
        self.log.info("✅ Awareness observer registered.")


    async def start_global_observation(self, client_id):
        """
        Observes awareness changes in global awarness
        """
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        collaboration = serverapp.web_app.settings["jupyter_server_ydoc"]
        websocket_server = collaboration.ywebsocket_server

        the_room_id = "JupyterLab:globalAwareness" 

        doc = websocket_server.rooms[the_room_id]

        async def on_awareness_change(event_type, data):
            doc = websocket_server.rooms[the_room_id]
            self.log.info(f"JUPYTER LAB GLOBAL: {doc.awareness.states}")
            self.log.info(f"CLIENT ID: {client_id}")

            global_doc = websocket_server.rooms[the_room_id]
            active_notebook_path = self.extract_current_notebook_path(global_doc, client_id)
            notebook = self.get_active_notebook(client_id, active_notebook_path)
            self.start_collaborative_session(notebook)
                
    
        awareness = doc.awareness
        awareness.observe(on_awareness_change)
        self.log.info("✅ GLOBAL Awareness observer registered.")



    async def stream_typing(self, full_text: str):
        stream_msg_id = self.ychat.add_message(NewMessage(body="", sender=self.id))
        current_text = ""

        for char in full_text:
            await asyncio.sleep(0.02)  # simulate typing speed
            current_text += char
            self.ychat.update_message(
                Message(
                    id=stream_msg_id,
                    body=current_text,
                    time=time(),
                    sender=self.id,
                    raw_time=False,
                ),
                append=False,  # we're replacing, not appending individual lines
            )



    async def process_message(self, message: Message):

        client_id = message.sender


        async def start_collaborative_session(): 
            await self.start_global_observation(client_id)

        # make a simple agent with one tool
        # that tool is starting a collaborative session


        self.log.info(f"MESSAGE BODY: {message.body}")
        if message.body == "@GrammarEditor Can you start a collaborative session?": 
            self._startCollab = True
            await self.stream_typing("Starting collaborative session now. Looking for grammar mistakes...")
            self.start_collaborative_session(notebook)




                
