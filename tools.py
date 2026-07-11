"""
tools.py
========
All the tools available to the agent live here. Keeping tools in their
own file (separate from the agent/model setup) makes it easy to add,
remove, or reuse tools across different agents.

To add a new tool: write a normal Python function with a clear docstring
and type hints, decorate it with @tool, then add it to the `all_tools`
list at the bottom of this file.
"""

from smolagents import tool, DuckDuckGoSearchTool


@tool
def calculator(expression: str) -> str:
    """
    Evaluates a basic math expression and returns the result.

    Args:
        expression: A math expression as a string, e.g. "23 * 7 + 1".

    Returns:
        The result of the calculation as a string.
    """
    try:
        # NOTE: eval() is used here only for a simple beginner demo.
        # In production code, use a safe math parser (e.g. the `numexpr`
        # or `asteval` library) instead of eval().
        allowed_chars = set("0123456789+-*/(). ")
        if not set(expression) <= allowed_chars:
            return "Error: expression contains disallowed characters."
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def word_counter(text: str) -> str:
    """
    Counts the number of words in a piece of text.

    Args:
        text: The text to count words in.

    Returns:
        A string reporting the word count.
    """
    count = len(text.split())
    return f"The text contains {count} words."


# DuckDuckGoSearchTool is a ready-made tool included with smolagents.
# It lets the agent search the live web and read back result snippets --
# no API key required.
web_search = DuckDuckGoSearchTool()


# Collect every tool here. Import `all_tools` wherever you build an agent
# so all your agents automatically share the same toolbox.
all_tools = [calculator, word_counter, web_search]