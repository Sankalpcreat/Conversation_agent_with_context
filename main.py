import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from utils.memory_handler import trim_messages, summarize_history


os.environ["OPENAI_API_KEY"]=""

model=ChatOpenAI(model="gpt-4o-mini")

workflow=StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
  
    system_prompt = (
        "You are a helpful assistant. Answer all questions to the best of your ability. "
        "The provided chat history includes a summary of the earlier conversation."
    )
    system_message = SystemMessage(content=system_prompt)

    message_history=state["messages"][:-1]

    trimmed_messages = trim_messages(message_history)

    
    user_message = state["messages"][-1]

    if len(trimmed_messages)>=4:

        summary_message=summarize_history(trimmed_messages,model)

        delete_message=[RemoveMessage(id=m.id) for m in state["messages"]]

        response=model.invoke([system_message,summary_message,user_message])

        message_update=[summary_message,user_message,response]+delete_message
    else:
        response=model.invoke([system_message]+trimmed_messages+[user_message])

        message_update=state["messages"] +[response]

    return {"messages":message_update}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")


memory_saver = MemorySaver()
app = workflow.compile(checkpointer=memory_saver)

if __name__=="__main__":
    user_id=input("Enter the User ID:") or "default_user"

    print("Start Chatting witht the assistant (type'exit' to stop):")

    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Chat ended.")
            break
        user_message = HumanMessage(content=user_input)

        response=app.invoke(
            {"messages":[user_message]},
            config={"configurable": {"thread_id": user_id}}
        )
        assistant_response = response["messages"][-1].content
        print(f"Assistant: {assistant_response}")




