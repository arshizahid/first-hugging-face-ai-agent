"""
Beginner-Friendly End-to-End AI Agent using Hugging Face
===========================================================

This script builds AI AGENTS using Hugging Face's `smolagents` library.
An "agent" here means: an LLM that can THINK, choose a TOOL to use,
ACT (run the tool), OBSERVE the result, and repeat until it has a final
answer -- instead of just replying with plain text.

This version:
  - Loads tools from a separate tools.py file
  - Includes a real web search tool (DuckDuckGoSearchTool)
  - Lets you compare two agent styles: CodeAgent vs ToolCallingAgent
  - Lets you easily switch between a big model and a smaller/faster one

------------------------------------------------------------------
STEP 0: INSTALL DEPENDENCIES
------------------------------------------------------------------
Run this in your terminal (not in this script) before running the file:

    pip install smolagents requests huggingface_hub ddgs

You will also need a FREE Hugging Face account + access token:
    1. Create an account at https://huggingface.co/join
    2. Get a token at https://huggingface.co/settings/tokens (read access is enough)
    3. Set it as an environment variable so the script can find it:

    On Mac/Linux:   export HF_TOKEN="your_token_here"
    On Windows:     set HF_TOKEN=your_token_here

Make sure tools.py is in the same folder as this file.
------------------------------------------------------------------
"""

import os
from smolagents import CodeAgent, ToolCallingAgent, InferenceClientModel

from tools import all_tools


# ------------------------------------------------------------------
# STEP 1: CONNECT TO A HUGGING FACE MODEL
# ------------------------------------------------------------------
# InferenceClientModel calls a Hugging Face-hosted model via Inference
# Providers, so you don't need a GPU or to download anything huge.

# IMPORTANT: Hugging Face's free Inference API was replaced by
# "Inference Providers" -- a marketplace of partners (Cerebras, Together,
# Fireworks, DeepInfra, etc). A model ID only works here if at least one
# partner is CURRENTLY hosting it -- and that list changes over time, so
# hardcoding a model name is fragile. Instead, we check a short list of
# candidates at runtime via model_info() and use the first one that's
# actually "warm" (live) right now.
#
# BIG candidates: larger / more capable, good for multi-step reasoning
BIG_CANDIDATES = [
    "Qwen/Qwen2.5-72B-Instruct",
    "openai/gpt-oss-120b",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-V3-0324",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
]
# SMALL candidates: smaller / faster / cheaper, good for simple tool calls
SMALL_CANDIDATES = [
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct-1M",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def find_working_model(candidates: list[str], hf_token: str) -> str:
    """
    Returns the first model in `candidates` that actually responds to a
    real request right now. We don't rely on model_info()'s "warm" flag
    alone -- it can say a model is warm even when the router rejects it --
    so we send a tiny live test message to confirm.
    """
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=hf_token)

    for model_id in candidates:
        try:
            client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=2,
            )
            return model_id
        except Exception:
            continue

    raise RuntimeError(
        "None of the candidate models are currently available via "
        "Inference Providers. Browse live models and pick one yourself at: "
        "https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation"
    )


def build_model(candidates: list[str] = SMALL_CANDIDATES):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "No Hugging Face token found. Set the HF_TOKEN environment "
            "variable before running this script (see Step 0 above)."
        )

    model_id = find_working_model(candidates, hf_token)
    print(f"Using model: {model_id}")

    model = InferenceClientModel(
        model_id=model_id,
        token=hf_token,
    )
    return model


# ------------------------------------------------------------------
# STEP 2: BUILD THE AGENTS
# ------------------------------------------------------------------
# CodeAgent: the agent writes small Python snippets to call tools and
#            reason step by step. Great for multi-step / math-heavy tasks,
#            and easy to debug since you see the actual code it runs.
#
# ToolCallingAgent: the agent picks a tool and arguments via structured
#            JSON-style calls (closer to "classic" function calling).
#            Often simpler and more predictable for single-tool lookups.

def build_code_agent(candidates: list[str] = SMALL_CANDIDATES):
    model = build_model(candidates)
    return CodeAgent(
        tools=all_tools,
        model=model,
        max_steps=6,
        verbosity_level=1,
    )


def build_tool_calling_agent(candidates: list[str] = SMALL_CANDIDATES):
    model = build_model(candidates)
    return ToolCallingAgent(
        tools=all_tools,
        model=model,
        max_steps=6,
        verbosity_level=1,
    )


# ------------------------------------------------------------------
# STEP 3: RUN IT END-TO-END
# ------------------------------------------------------------------
if __name__ == "__main__":
    question = (
        "Search the web for the current population of Japan, then tell me "
        "how many words are in your answer sentence."
    )

    print(f"\nUser question: {question}\n")

    # --- Run with CodeAgent ---
    print("=" * 60)
    print("Running with CodeAgent (writes Python to call tools)")
    print("=" * 60)
    code_agent = build_code_agent(BIG_CANDIDATES)
    code_answer = code_agent.run(question)
    print("\nCodeAgent FINAL ANSWER:")
    print(code_answer)

    # --- Run with ToolCallingAgent ---
    print("\n" + "=" * 60)
    print("Running with ToolCallingAgent (structured JSON tool calls)")
    print("=" * 60)
    tool_agent = build_tool_calling_agent(BIG_CANDIDATES)
    tool_answer = tool_agent.run(question)
    print("\nToolCallingAgent FINAL ANSWER:")
    print(tool_answer)

    # To use the bigger, more capable model instead, just pass:
    #   build_code_agent(BIG_CANDIDATES)
    #   build_tool_calling_agent(BIG_CANDIDATES)