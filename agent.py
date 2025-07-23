import functools
import os
from typing import Any, Generator, Literal, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    UCFunctionToolkit,
)
from databricks_langchain.genie import GenieAgent
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from pydantic import BaseModel

###################################################
## Create a GenieAgent with access to a Genie Space
###################################################

# TODO add GENIE_SPACE_ID and a description for this space
# You can find the ID in the URL of the genie room /genie/rooms/<GENIE_SPACE_ID>
# Example description: This Genie agent can answer questions based on a database containing tables related to enterprise software sales, including accounts, opportunities, opportunity history, fiscal periods, quotas, targets, teams, and users. Use Genie to fetch and analyze data from these tables by specifying the relevant columns and filters. Genie can execute SQL queries to provide precise data insights based on your questions.
GENIE_SPACE_ID = "01f0678babbe18fdbbd7729053d07ab9" # The one after
genie_agent_description = "This is a Genie agent that you can converse with to get answers to questions. Try to provide simple questions and provide history if you had prior conversations. Use this tool to query HL7 patient data, focus on finding allergy data for a patient.  Do not use the Genie agent to get ICD-10 definition.  AL1_2 is not ICD-10 code.  Don't confused AL1_2 with ICD-10 code."

genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="Genie",
    description=genie_agent_description,
    client=WorkspaceClient(
        host=os.getenv("DB_MODEL_SERVING_HOST_URL"),
        token=os.getenv("DATABRICKS_GENIE_PAT"),
    ),
)


############################################
# Define your LLM endpoint and system prompt
############################################

# TODO: Replace with your model serving endpoint
# multi-agent Genie works best with claude 3.7 or gpt 4o models.
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)


############################################################
# Create a code agent
# You can also create agents with access to additional tools
############################################################
tools = []

# Parameterize the vector search index
catalog_name = "danny_catalog"
schema_name = "clinical_coding"
vs_index_name = "icd10_docs_chunked_index"
vs_index_full_name = f"{catalog_name}.{schema_name}.{vs_index_name}"

# TODO if desired, add additional tools and update the description of this agent
# uc_tool_names = ["system.ai.*"]
# uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# tools.extend(uc_toolkit.tools)
# code_agent_description = (
#     "The Coder agent specializes in solving programming challenges, generating code snippets, debugging issues, and explaining complex coding concepts.",
# )
# code_agent = create_react_agent(llm, tools=tools)

vs_agent_description = (
    "The Vector Search agent specializes in looking into ICD-10 definitions from unstructured text stored in Vector Search.  For questions that related to ICD-10 clinical coding, use this tool.",
)
vector_search_tools = [
        VectorSearchRetrieverTool(
        index_name=vs_index_full_name,
        # filters="..."
    )
]
tools.extend(vector_search_tools)

vs_agent = create_react_agent(llm, tools=tools)

#############################
# Define the supervisor agent
#############################

# TODO update the max number of iterations between supervisor and worker nodes
# before returning to the user
MAX_ITERATIONS = 3

worker_descriptions = {
    "Genie": genie_agent_description,
    "VectorSearch": vs_agent_description,
}

formatted_descriptions = "\n".join(
    f"- {name}: {desc}" for name, desc in worker_descriptions.items()
)

system_prompt = f"You are analysing the HL7 patient data with allergy.  Get the patient data with the genie_query tool.  Focus on segment_id equal to AL1.  Focus on the column AL1_3 only, ignore column AL1_2, which is the description of the allergy.  Use vector search tool to get the ICD10 text chunk for ICD10 clinical code identification based on the allergy description.  Decide between routing between the following workers or ending the conversation if an answer is provided. \n{formatted_descriptions}"
options = ["FINISH"] + list(worker_descriptions.keys())
FINISH = {"next_node": "FINISH"}

def supervisor_agent(state):
    count = state.get("iteration_count", 0) + 1
    if count > MAX_ITERATIONS:
        return FINISH
    
    class nextNode(BaseModel):
        next_node: Literal[tuple(options)]

    preprocessor = RunnableLambda(
        lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    supervisor_chain = preprocessor | llm.with_structured_output(nextNode)
    next_node = supervisor_chain.invoke(state).next_node
    
    # if routed back to the same node, exit the loop
    if state.get("next_node") == next_node:
        return FINISH
    return {
        "iteration_count": count,
        "next_node": next_node
    }

#######################################
# Define our multiagent graph structure
#######################################


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [
            {
                "role": "assistant",
                "content": result["messages"][-1].content,
                "name": name,
            }
        ]
    }


def final_answer(state):
    prompt = "Using only the content in the messages, respond to the previous user question using the answer given by the other assistant messages."
    preprocessor = RunnableLambda(
        lambda state: state["messages"] + [{"role": "user", "content": prompt}]
    )
    final_answer_chain = preprocessor | llm
    return {"messages": [final_answer_chain.invoke(state)]}


class AgentState(ChatAgentState):
    next_node: str
    iteration_count: int


vs_node = functools.partial(agent_node, agent=vs_agent, name="VectorSearch")
genie_node = functools.partial(agent_node, agent=genie_agent, name="Genie")

workflow = StateGraph(AgentState)
workflow.add_node("Genie", genie_node)
workflow.add_node("VectorSearch", vs_node)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("final_answer", final_answer)

workflow.set_entry_point("supervisor")
# We want our workers to ALWAYS "report back" to the supervisor when done
for worker in worker_descriptions.keys():
    workflow.add_edge(worker, "supervisor")

# Let the supervisor decide which next node to go
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next_node"],
    {**{k: k for k in worker_descriptions.keys()}, "FINISH": "final_answer"},
)
workflow.add_edge("final_answer", END)
multi_agent = workflow.compile()

###################################
# Wrap our multi-agent in ChatAgent
###################################


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg})
                    for msg in node_data.get("messages", [])
                )


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
mlflow.langchain.autolog()
AGENT = LangGraphChatAgent(multi_agent)
mlflow.models.set_model(AGENT)
