###############################################################################
# 1. Setup: Configure Vertex AI and the Gemini model
###############################################################################
import vertexai
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    Part,
    FunctionDeclaration,
    Tool,
)
import pandas as pd
import json
from IPython.display import Javascript, display, Markdown
from IPython import get_ipython
import io
import sys

# Initialize your environment
PROJECT_ID = "your-project-id"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)

model = GenerativeModel("gemini-1.5-flash-002")

###############################################################################
# 2. Define a Function & Tool for getting JSON steps
###############################################################################
# Example JSON schema for a "get_steps_plan" function that returns a list of steps
get_steps_plan_func = FunctionDeclaration(
    name="get_steps_plan",
    description="Get a plan in JSON with multiple exploration steps.",
    parameters={
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": "An array of step descriptions for the data exploration plan.",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["steps"]
    },
)

# We'll add the function to a Tool object so the model can call it
plan_tool = Tool(function_declarations=[get_steps_plan_func])

###############################################################################
# 3. LLM function using the Gemini model with function calls
###############################################################################
def call_llm_for_plan(user_prompt):
    """
    Calls the Gemini model specifying that it should respond by calling
    the 'get_steps_plan' function, returning JSON with steps.
    """
    # Construct the conversation content
    conversation = Content(
        parts=[
            Part(
                role="system",
                text=(
                    "You are an AI assistant that provides data exploration steps "
                    "as JSON, conforming to the function signature defined for you."
                ),
            ),
            Part(
                role="user",
                text=user_prompt,
            ),
        ]
    )

    # Make the request, including the 'plan_tool'
    response = model.generate_content(
        conversation,
        generation_config=GenerationConfig(temperature=0.0),
        tools=[plan_tool],
    )
    
    # Parse any function call results. If the model calls our function,
    # it will appear in `response.tool_calls`.
    if response.tool_calls:
        # Normally there's only one relevant function call in this scenario
        for tool_call in response.tool_calls:
            if tool_call.function_name == "get_steps_plan":
                # The arguments are already in JSON format
                return tool_call.arguments
    # If no function call was made or something unexpected, return None or empty
    return None

###############################################################################
# 4. Helper / Utility functions (same as your original code)
###############################################################################
def clean_code(code):
    """Strip out Markdown fences from generated code."""
    return code.replace("```python", "").replace("```", "")

def remove_duplicate_imports(code):
    """Skip duplicate import lines from each code snippet."""
    lines = []
    for line in code.splitlines():
        if line.strip().startswith("import "):
            continue
        lines.append(line)
    return "\n".join(lines)

def run_code_in_backend(code):
    """Runs code in the current Python kernel and captures any printed output."""
    ipy = get_ipython()
    backup_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured
    try:
        ipy.run_cell(code, store_history=False)
    except Exception as e:
        print(f"[Error while running code]: {e}")
    finally:
        sys.stdout = backup_stdout
    return captured.getvalue()

def create_code_cell_visually(code):
    """Creates and auto-runs a new Jupyter code cell with the given code text."""
    escaped_code = code.replace("`", "\\`")
    display(Javascript(f"""
        var code = `{escaped_code}`;
        var cell = Jupyter.notebook.insert_cell_below('code');
        cell.set_text(code);
        // Optionally auto-run the new cell
        Jupyter.notebook.select(cell);
        Jupyter.notebook.execute_cell_and_select_below();
    """))

def create_and_run_cell(code):
    """Cleans up, executes code in the backend, and creates a new cell in the UI."""
    final_code = remove_duplicate_imports(code)
    output = run_code_in_backend(final_code)
    create_code_cell_visually(final_code)
    return output

###############################################################################
# 5. Main automation function for multi-DataFrame exploration
###############################################################################
def auto_explore_dataframes(dataframes, user_goal, llm_plan_function):
    """
    Automates multi-step data exploration with multiple DataFrames, using the
    function-based approach to get well-structured JSON steps.
    """
    # Summaries for context
    df_summaries = []
    for name, df in dataframes.items():
        df_summaries.append(
            f"- **{name}**: columns={list(df.columns)};\n"
            f"  Sample:\n  {df.head(2).to_dict()}\n"
        )
    combined_df_info = "\n".join(df_summaries)

    # --------------------------------------------------------------------------
    # A) Prompt the LLM for a plan using a function call that returns { steps: [...] }
    # --------------------------------------------------------------------------
    plan_prompt = f"""
We have multiple DataFrames loaded in memory:
{combined_df_info}

The user wants to accomplish:
{user_goal}

Produce an array of steps under a JSON key "steps" without extra commentary.
Do not recreate or redefine any DataFrames.
"""
    plan_data = llm_plan_function(plan_prompt)
    if plan_data is None:
        print("No valid steps returned from the LLM plan function.")
        return

    # plan_data should be a string of JSON, e.g. '{"steps": ["Step 1", "Step 2"]}'
    try:
        plan_json = json.loads(plan_data)
        steps = plan_json.get("steps", [])
    except json.JSONDecodeError:
        print("[Plan Parsing] Invalid JSON, skipping steps.")
        steps = []

    # --------------------------------------------------------------------------
    # B) Execute each step by requesting code from the LLM (or from the same Gemini model)
    #    For brevity, let's assume we use a simpler prompt-based approach or define
    #    another function-based method for code generation.
    # --------------------------------------------------------------------------
    all_outputs = []
    for idx, step in enumerate(steps, start=1):
        code_prompt = f"""
We have DataFrames in memory: {list(dataframes.keys())}
(They are already defined; do NOT recreate them.)

Step to accomplish:
"{step}"

Return Python code only (no extra text) for a Jupyter cell 
that achieves this step (e.g., prints stats, calls .describe(), etc.).
"""
        # Reuse the same model but with a simpler prompt (no function needed here)
        conversation = Content(
            parts=[
                Part(
                    role="system",
                    text=(
                        "You are an AI assistant that produces pure Python code for data analysis. "
                        "Return code only. No extra commentary."
                    ),
                ),
                Part(role="user", text=code_prompt),
            ]
        )
        step_response = model.generate_content(
            conversation, 
            generation_config=GenerationConfig(temperature=0.0)
        )
        
        generated_code = clean_code(step_response.text)
        output_str = create_and_run_cell(generated_code)
        all_outputs.append((step, output_str))

    # --------------------------------------------------------------------------
    # C) Potentially ask if more steps are needed (with another function-based approach)
    # --------------------------------------------------------------------------
    # (Optional) You can define another function schema for "check_followup" to see if 
    # more steps are required, etc.

    # --------------------------------------------------------------------------
    # D) Summarize
    # --------------------------------------------------------------------------
    summary_prompt = f"""
Below is a list of steps executed and their corresponding outputs:
{all_outputs}

Please provide a concise summary of the key findings (plain text).
No JSON needed.
"""
    conversation = Content(
        parts=[
            Part(
                role="system",
                text="You are an AI assistant that provides a concise, plain-text summary."
            ),
            Part(
                role="user",
                text=summary_prompt
            ),
        ]
    )
    summary_response = model.generate_content(
        conversation,
        generation_config=GenerationConfig(temperature=0.0),
    )
    display(Markdown(f"## Final Conclusion & Findings\n\n{summary_response.text}"))

###############################################################################
# 6. Example usage
###############################################################################
if __name__ == "__main__":
    # Example DataFrame 1
    df_customers = pd.DataFrame({
        "CustomerID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "City": ["New York", "Los Angeles", "Chicago"]
    })

    # Example DataFrame 2
    df_orders = pd.DataFrame({
        "CustomerID": [1, 2, 3],
        "OrderID": [101, 102, 103],
        "Product": ["Widget A", "Widget B", "Widget C"],
        "Price": [19.99, 29.99, 9.99],
        "Quantity": [10, 5, 20]
    })

    dataframes_dict = {
        "df_customers": df_customers,
        "df_orders": df_orders
    }
    user_goal = "Explore the customers and their orders for some basic stats."

    # Run the auto-exploration with function-based plan retrieval
    auto_explore_dataframes(dataframes_dict, user_goal, call_llm_for_plan)
