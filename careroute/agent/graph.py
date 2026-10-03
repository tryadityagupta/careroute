"""
agent/graph.py — CareRoute's agent as an explicit LangGraph state machine.

        START
          |
          v
    +-> agent ---(tool calls, budget left)---> tools ---+
    |                                                   |
    +---------------------------------------------------+
          |
          |--(no tool calls)---> guard --> END    deterministic answer check
          |--(budget spent)----> stop  --> END    max-model-calls cap

Two compiled forms of the SAME graph:
  * single-shot (/care) — no checkpointer;
  * conversational (/chat) — with a checkpointer, keyed by thread_id (= the
    session id), so turn 2 on another replica sees turn 1.

`model` is a public attribute: anything with an async
.ainvoke(messages) -> AIMessage. Production passes ChatOpenAI.bind_tools(...);
tests pass a scripted model.

ASYNC end to end: every node is a coroutine, the model is awaited, tools are
awaited (concurrently when one reply asks for several), and the checkpointer
is AsyncPostgresSaver. A turn spends most of its life waiting on the model;
while it waits, the replica's event loop serves other turns — there is no
worker thread held per request any more.
"""

from __future__ import annotations

import asyncio
import time

from langchain_core.messages import (AIMessage, HumanMessage, RemoveMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from careroute.agent.checkpointer import CheckpointerFactory
from careroute.agent.guard import AnswerGuard
from careroute.agent.messages import names_in, payload_of, summarize, text_of
from careroute.agent.prompts import EMERGENCY_CONTEXT, SYSTEM_PROMPT
from careroute.agent.state import CareRouteState, fresh_conversation, fresh_turn
from careroute.agent.tracing import TurnTracer


class CareRouteAgent:
    STOPPED = "Stopped: reached the maximum number of steps without a final answer."

    def __init__(self, model, tools: list, *, checkpointers: CheckpointerFactory | None = None,
                 guard: AnswerGuard | None = None, tracer: TurnTracer | None = None,
                 max_llm_calls: int = 8):
        self.model = model
        self.max_llm_calls = max_llm_calls
        self.guard = guard or AnswerGuard()
        self.tracer = tracer or TurnTracer()
        self.checkpointers = checkpointers or CheckpointerFactory()
        self._tool_node = ToolNode(tools)       # runs tools, catches exceptions
        self._builder = self._build_graph()
        self.graph = self._builder.compile()
        self._conversation_graph = None
        self._conv_lock: asyncio.Lock | None = None

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        b = StateGraph(CareRouteState)
        b.add_node("agent", self._agent_node)
        b.add_node("tools", self._tools_node)
        b.add_node("guard", self._guard_node)
        b.add_node("stop", self._stop_node)
        b.add_edge(START, "agent")
        b.add_conditional_edges("agent", self._route_after_agent,
                                {"tools": "tools", "guard": "guard", "stop": "stop"})
        b.add_edge("tools", "agent")
        b.add_edge("guard", END)
        b.add_edge("stop", END)
        return b

    async def conversation_graph(self):
        """The graph compiled WITH a checkpointer, built on first use."""
        if self._conversation_graph is None:
            if self._conv_lock is None:
                self._conv_lock = asyncio.Lock()
            async with self._conv_lock:
                if self._conversation_graph is None:
                    self._conversation_graph = self._builder.compile(
                        checkpointer=await self.checkpointers.get())
        return self._conversation_graph

    @staticmethod
    def _compact(messages: list) -> list:
        """Shrink what is re-sent every call: tool results from EARLIER turns
        become a stub (the earlier answers already name the providers). The
        ToolMessages stay, so tool_call/tool_result pairing remains valid."""
        humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        last_human = humans[-1] if humans else 0
        return [m.model_copy(update={"content": "(earlier tool result omitted)"})
                if i < last_human and isinstance(m, ToolMessage) else m
                for i, m in enumerate(messages)]

    async def _agent_node(self, state: CareRouteState):
        """REASON: ask the model what to do next. The system prompt is added
        here, never stored, so a prompt edit can't break a saved conversation."""
        prompt = SYSTEM_PROMPT
        if state.get("emergency_number"):
            prompt += EMERGENCY_CONTEXT.format(number=state["emergency_number"])
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=prompt), *messages]
        t0 = time.perf_counter()
        # ainvoke: while the model thinks (seconds), this replica serves others.
        response = await self.model.ainvoke(self._compact(messages))
        n = state.get("llm_calls", 0) + 1
        print(f" [call {n}] LLM took {time.perf_counter() - t0:.1f}s")
        for call in response.tool_calls:
            print(f" [call {n}] model called: {call['name']}({call['args']})")
        return {"messages": [response], "llm_calls": n}

    async def _tools_node(self, state: CareRouteState, config: RunnableConfig):
        """ACT + OBSERVE, then sort every returned facility name into
        confirmed (a success list from find_providers / find_pharmacies) or
        offered (anything else). The guard is only as good as this sorting."""
        t0 = time.perf_counter()
        # config carries the patient binding. Several tool calls in one model
        # reply run CONCURRENTLY (ToolNode gathers async tools).
        result = await self._tool_node.ainvoke(state, config)
        print(f"          tools took {time.perf_counter() - t0:.1f}s")

        confirmed = set(state.get("confirmed_providers") or ())
        turn_confirmed = set(state.get("turn_confirmed") or ())
        offered = set(state.get("offered_facilities") or ())
        errors = list(state.get("tool_errors") or ())
        is_emergency = bool(state.get("is_emergency"))
        emergency_number = state.get("emergency_number") or ""
        for msg in result["messages"]:
            if not isinstance(msg, ToolMessage):
                continue
            payload = payload_of(msg)
            print(f"          -> {summarize(payload)}")
            if isinstance(payload, dict) and payload.get("emergency"):
                is_emergency = True
                emergency_number = payload.get("emergency_number") or emergency_number or "112"
            if isinstance(payload, dict) and "error" in payload:
                errors.append(f"{msg.name}: {payload['error']}")
            elif msg.name == "find_providers" and isinstance(payload, list):
                turn_confirmed |= names_in(payload)
            elif (msg.name == "find_pharmacies" and isinstance(payload, dict)
                  and payload.get("pharmacies")):
                # A pharmacy the tool returned IS the confirmed answer to
                # "where do I buy this".
                turn_confirmed |= names_in(payload)
            else:
                offered |= names_in(payload)
        return {"messages": result["messages"],
                "confirmed_providers": confirmed | turn_confirmed,
                "turn_confirmed": turn_confirmed,
                "offered_facilities": offered,
                "tool_errors": errors,
                "is_emergency": is_emergency,
                "emergency_number": emergency_number}

    async def _guard_node(self, state: CareRouteState):
        """If the answer must change, REMOVE the model's message and append the
        guarded one, so the transcript never keeps the untrusted version."""
        final = state["messages"][-1]
        answer = text_of(final)
        guarded = self.guard.review(answer, state)
        if guarded == answer:
            return {}
        swap = [AIMessage(content=guarded)]
        if final.id:
            swap.insert(0, RemoveMessage(id=final.id))
        return {"messages": swap}

    async def _stop_node(self, state: CareRouteState):
        """Budget spent mid-plan. The dangling assistant message still REQUESTS
        tools; leaving it would corrupt the transcript (OpenAI rejects a tool
        call with no result), so it is replaced by a plain statement."""
        dangling = state["messages"][-1]
        swap = [AIMessage(content=self.STOPPED)]
        if dangling.id:
            swap.insert(0, RemoveMessage(id=dangling.id))
        return {"messages": swap}

    def _route_after_agent(self, state: CareRouteState) -> str:
        last = state["messages"][-1]
        if not last.tool_calls:
            return "guard"                   # model is done -> validate the answer
        if state["llm_calls"] >= self.max_llm_calls:
            return "stop"                    # wants more tools, out of budget
        return "tools"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @staticmethod
    def _config(patient_id: str | None, thread_id: str | None = None) -> dict:
        configurable = {}
        if thread_id:
            configurable["thread_id"] = thread_id
        if patient_id:
            configurable["patient_id"] = patient_id
        return {"configurable": configurable}

    def _finish(self, final_state: dict) -> tuple[str, dict]:
        msgs = final_state["messages"]
        trace = self.tracer.trace(msgs)
        trace["llm_calls"] = final_state.get("llm_calls", trace["llm_calls"])
        return text_of(msgs[-1]), trace

    async def run_single(self, user_request: str, *,
                         patient_id: str | None = None) -> tuple[str, dict]:
        """One self-contained request (/care). Returns (answer, trace)."""
        final = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=user_request)], **fresh_conversation()},
            config=self._config(patient_id))
        return self._finish(final)

    async def run_turn(self, user_message: str, *, thread_id: str,
                       patient_id: str | None = None,
                       new_conversation: bool | None = None) -> tuple[str, dict]:
        """One turn of a conversation (/chat). Returns (answer, trace).

        new_conversation: pass it when you know (the server does: turn 1 vs
        later) to skip a checkpoint read. None = ask the checkpointer.
        """
        conv = await self.conversation_graph()
        config = self._config(patient_id, thread_id)
        if new_conversation is None:
            try:
                new_conversation = not (await conv.aget_state(config)).values
            except Exception:
                new_conversation = True
        turn = {"messages": [HumanMessage(content=user_message)],
                **(fresh_conversation() if new_conversation else fresh_turn())}
        # durability="exit": ONE checkpoint when the turn ends instead of one
        # per graph step (~5-10x fewer writes). If a replica dies mid-turn, the
        # conversation resumes from the previous turn — what a user expects
        # after an error anyway.
        final = await conv.ainvoke(turn, config=config, durability="exit")
        return self._finish(final)
