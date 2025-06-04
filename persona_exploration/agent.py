# langgraph_agent_runner.py

from typing import Dict, Any, Optional, Tuple
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from jupyterlab_chat.models import Message, NewMessage

import json
import asyncio
from typing_extensions import TypedDict
from jupyter_server_ai_tools import run_tools
from time import time


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

async def run_langgraph_agent(logger, ychat, self_id, extension_manager, notebook, user_prompt: str, tone_prompt: str, tool_groups):
    memory = MemorySaver()
    tools = [t["metadata"] for t in tool_groups.values()]
    openai_functions = convert_mcp_to_openai(tools)
    llm = ChatOpenAI(temperature=0, streaming=True).bind_tools(tools=openai_functions)

    def parse_openai_tool_call(call: Dict) -> Tuple[str, Dict]:
        fn = call.get("function", {})
        name = fn.get("name")
        arguments = fn.get("arguments", "{}")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        logger.info(f"TOOL NAME: {name}")
        if not name.startswith("git_"):
            logger.info("📎 Injecting ynotebook into tool arguments")
            arguments["ynotebook"] = notebook

        return name, arguments

    async def agent(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        stream_message_id = None
        full_chunk: Optional[AIMessageChunk] = None

        async for chunk in llm.astream(messages):
            if full_chunk is None:
                full_chunk = chunk
            else:
                full_chunk += chunk
            content = getattr(chunk, "content", "")
            if content:
                if stream_message_id is None:
                    stream_message_id = ychat.add_message(NewMessage(body=content, sender=self_id))
                else:
                    ychat.update_message(Message(id=stream_message_id, body=content, time=time(), sender=self_id, raw_time=False), append=True)

        full_message = AIMessage(content=full_chunk.content if full_chunk else "",  additional_kwargs=full_chunk.additional_kwargs if full_chunk else {},
            response_metadata=full_chunk.response_metadata if full_chunk else {})
        logger.info("✅ full_message: %s", full_message)
        return {"messages": messages + [full_message]}
    
    def should_continue(state):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and "tool_calls" in last_message.additional_kwargs:
            logger.info("CONTINUING ------------------------------")
            return "continue"
        if isinstance(last_message, ToolMessage):
            logger.info("CONTINUING -------------------------------")
            return "continue"
        logger.info("ENDING ----------------------------------")
        return "end"

    async def call_tool(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        last_msg = messages[-1]
        tool_calls = last_msg.additional_kwargs.get("tool_calls", [])
        if len(tool_calls) > 1:
            logger.warning("⚠️ Multiple tool calls detected. Only executing the first.")
            tool_calls = [tool_calls[0]]

        results = []

        for call in tool_calls:
            tool_name = call["function"]["name"]
            calling_msg = f"🔧 Calling {tool_name}...\n"
            stream_msg_id = ychat.add_message({"body": "", "sender": self_id})
            for char in calling_msg:
                await asyncio.sleep(0.01)
                ychat.update_message({"id": stream_msg_id, "body": char, "sender": self_id}, append=True)

            result = await run_tools(extension_manager, [call], parse_fn=parse_openai_tool_call)
            logger.info(f"TOOL RESULTS: {result}")
            tool_result = result[0]
            if asyncio.iscoroutine(tool_result):
                logger.warning("⚠️ Tool returned a coroutine — awaiting it before serialization.")
                tool_result = await tool_result

            tool_output = {"result": str(tool_result)}
            results.append((call, tool_output))

        tool_messages = [
            ToolMessage(
                name=call["function"]["name"],
                tool_call_id=call["id"],
                content=json.dumps(result_dict),
            )
            for call, result_dict in results
        ]
        return {"messages": messages + tool_messages}

    workflow = StateGraph(State)
    workflow.add_node("agent", agent)
    workflow.add_node("call_tool", call_tool)
    workflow.set_entry_point("agent")
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "call_tool", "end": END})
    workflow.add_edge("call_tool", "agent")

    compiled = workflow.compile(checkpointer=memory)

    system_prompt = f"""
        You are a function-calling assistant operating inside a JupyterLab environment.
        Your job is to read the entire notebook and if you find grammar mistakes in markdown cells, 
        and fix them using your available tools. Only operate on markdown cells please. 
        Along with fixing grammatical mistakes, please adapt markdown text to the following tone: {tone_prompt}
        You may only call one tool at a time. If you want to perform multiple actions, wait for confirmation and state each one step by step.
        Please focus on tool calls and not sending messages to the user. 
        Please start by calling read_notebook.
        """
    logger.info(f"PROMPT: {system_prompt}")

    state = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    config = {"configurable": {"thread_id": "thread-1"}}
    logger.info("HERE")
    await compiled.ainvoke(state, config=config)



async def run_supervisor_agent(logger, stream, user_message: str, tools):
    memory = MemorySaver()
    llm = ChatOpenAI(temperature=0, streaming=True).bind_tools(tools=tools)

    async def agent(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        stream_message_id = None
        full_chunk: Optional[AIMessageChunk] = None

        response = await llm.ainvoke(messages)
        await stream(response.content or "")


        return {"messages": messages + [response]}
    
    def should_continue(state):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and "tool_calls" in last_message.additional_kwargs:
            logger.info("CONTINUING ------------------------------")
            return "continue"
        if isinstance(last_message, ToolMessage):
            logger.info("CONTINUING -------------------------------")
            return "continue"
        logger.info("ENDING ----------------------------------")
        return "end"
    
    async def call_tool(state: Dict[str, Any]) -> Dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_calls = last_msg.additional_kwargs.get("tool_calls", [])
        results = []

        for call in tool_calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            logger.info(f"Calling tool: {name} with args: {args}")

            # Look up tool by name
            for tool in tools:
                if tool.name == name:
                    result = await tool.ainvoke(args)
                    await stream(f"🔧 {name} executed: {result}")
                    tool_message = ToolMessage(
                        tool_call_id=call["id"],
                        name=name,
                        content=json.dumps({"result": result})
                    )
                    results.append(tool_message)
                    break
            else:
                logger.warning(f"No tool found for {name}")

        return {"messages": state["messages"] + results}



    workflow = StateGraph(State)
    workflow.add_node("agent", agent)
    workflow.add_node("call_tool", call_tool)
    workflow.set_entry_point("agent")
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "call_tool", "end": END})
    workflow.add_edge("call_tool", "agent")

    compiled = workflow.compile(checkpointer=memory)


    system_prompt = """
        You are a helpful assistant that can start or stop a collaborative grammar editing session.
        The user may describe their tone preferences for the way they want their markdown cells edited. 
        """
        

    state = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    }
    config = {"configurable": {"thread_id": "thread-1"}}
    logger.info("HERE")
    await compiled.ainvoke(state, config=config)