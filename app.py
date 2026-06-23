import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Setup
# =========================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# =========================
# Tools
# =========================

def get_weather(city: str):
    url = f"https://wttr.in/{city}?format=%C+%t"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            return f"The weather in {city} is {response.text}"

        return "Unable to fetch weather."

    except Exception as e:
        return str(e)


def run_command(cmd: str):
    result = os.system(cmd)
    return str(result)


def get_stock_price(ticker: str):
    try:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        url = "https://www.alphavantage.co/query"

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker.upper(),
            "apikey": api_key
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        quote = data.get("Global Quote", {})

        if not quote:
            return f"Could not fetch price for {ticker}"

        price = float(quote["05. price"])

        change = quote["09. change"]
        change_pct = quote["10. change percent"]

        return (
            f"The current price of {ticker.upper()} "
            f"is ${price:.2f} | "
            f"Change: {change} ({change_pct})"
        )

    except Exception as e:
        return str(e)


available_tools = {
    "get_weather": get_weather,
    "run_command": run_command,
    "get_stock_price": get_stock_price,
}

# =========================
# System Prompt
# =========================

SYSTEM_PROMPT = """
You are a helpful AI Assistant who is specialized in resolving user queries.

You work on:
start -> plan -> action -> observe -> output

Rules:
- Follow JSON output format
- Perform one step at a time
- Think step by step

Output JSON Format:

{
    "step":"string",
    "content":"string",
    "function":"tool_name",
    "input":"tool_input"
}

Available Tools:

1. get_weather(city)
2. run_command(command)
3. get_stock_price(ticker)
"""

# =========================
# Session State
# =========================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# UI
# =========================

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Agent with Tools")

# Show chat history

for msg in st.session_state.chat_history:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# Chat Input
# =========================

user_query = st.chat_input("Ask me anything...")

if user_query:

    # Display user message

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("assistant"):

        placeholder = st.empty()

        final_answer = ""

        while True:

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                messages=st.session_state.messages
            )

            assistant_response = (
                response.choices[0]
                .message
                .content
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_response
                }
            )

            parsed_response = json.loads(
                assistant_response
            )

            step = parsed_response.get("step")

            # PLAN

            if step == "plan":

                plan_text = (
                    "🧠 "
                    + parsed_response.get("content")
                )

                placeholder.info(plan_text)

                continue

            # ACTION

            elif step == "action":

                tool_name = parsed_response.get(
                    "function"
                )

                tool_input = parsed_response.get(
                    "input"
                )

                placeholder.warning(
                    f"🛠️ Calling Tool: "
                    f"{tool_name}"
                    f"({tool_input})"
                )

                if tool_name in available_tools:

                    output = available_tools[
                        tool_name
                    ](tool_input)

                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "step": "observe",
                                    "output": output
                                }
                            )
                        }
                    )

                continue

            # OUTPUT

            elif step == "output":

                final_answer = parsed_response.get(
                    "content"
                )

                placeholder.success(final_answer)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": final_answer
                    }
                )

                break