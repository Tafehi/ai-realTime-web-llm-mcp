from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph
from chains.chain import Chain
from tools.serpApi import SerpApiSearch


class Graph:
    def __init__(self, text):
        self.chain = Chain()
        self.text = text
        self.graph = None
        self.__max_iterations = 2

    def build_chain(self):
        builder = MessageGraph()
        chains = self.chain
        serp = SerpApiSearch()

        # Add nodes
        builder.add_node("draft", chains.first_response())
        builder.add_node("serp_tool", serp.search_serpapi)
        builder.add_node("revised", chains.revision_response())

        # Set entry point
        builder.set_entry_point("draft")

        # Add edges
        builder.add_edge("draft", "serp_tool")
        builder.add_edge("serp_tool", "revised")

        # Add conditional edge from "revised" to either END or "serp_tool"
        builder.add_conditional_edges(
            "revised", self.revisory_node, {END: END, "serp_tool": "serp_tool"}
        )

        self.graph = builder.compile()
        # print(self.graph.get_graph().draw_mermaid())
        return self.graph

    # This function decides whether to continue or stop
    def revisory_node(self, state: List[BaseMessage]) -> str:
        count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
        if count_tool_visits > self.__max_iterations:
            return END
        return "serp_tool"
