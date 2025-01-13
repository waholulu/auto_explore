###############################################################################
# 0. Dependencies: Install and import Vertex AI Libraries
#    (e.g., pip install google-cloud-aiplatform)
###############################################################################
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

import pandas as pd
import json
import io
import sys

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

###############################################################################
# 1. Setup: Configure your Vertex AI Project / Location and Initialize Model
###############################################################################

# TODO: Replace with your actual project ID and desired location
PROJECT_ID = "YOUR_PROJECT_ID"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Load the Gemini model
model = GenerativeModel("gemini-1.5-pro-002")


###############################################################################
# 2. Define JSON schemas for the responses
###############################################################################
# A) For the plan or follow-up steps, which look like:
# {
#   "steps": ["Step 1 description", "Step 2 description", ...]
# }
plan_steps_schema = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["steps"],
}


###############################################################################
# 3. LLM wrapper functions for different response types
###############################################################################
def call_llm_for_json_steps(prompt, temperature=0.0):
    """
    Calls Vertex AI Gemini model for a strictly JSON array-of-strings (plan steps).
    """
    system_instructions = (
        "You are an AI assistant that provides helpful, "
        "accurate Python code for data exploration tasks. "
        "Return only valid JSON if requested."
    )

    combined_prompt = system_instructions + "\n" + prompt

    generation_config = GenerationConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=plan_steps_schema
    )

    try:
        response = model.generate_content(
            combined_prompt,
            generation_config=generation_config
        )
        return response.text  # Should be valid JSON matching the schema
    except Exception as e:
        print(f"[LLM Error]: {e}")
        return ""


def call_llm_for_text(prompt, temperature=0.0):
    """
    Calls Vertex AI Gemini model for a plain text response (e.g., final summary).
    """
    system_instructions = (
        "You are an AI assistant that provides helpful, "
        "accurate Python code for data exploration tasks."
    )

    combined_prompt = system_instructions + "\n" + prompt

    generation_config = GenerationConfig(
        temperature=temperature,
        # We'll let the model return plain text for final summaries
        response_mime_type="text/plain"
    )

    try:
        response = model.generate_content(
            combined_prompt,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        print(f"[LLM Error]: {e}")
        return ""


###############################################################################
# 4. Helper / Utility functions
###############################################################################
def clean_code(code):
    """
    Strip out Markdown fences from generated code for safe execution.
    """
    return (
        code.replace("```python", "")
            .replace("```", "")
    )

def remove_duplicate_imports(code):
    """
    For brevity, skip duplicate import lines from each code snippet.
    """
    lines = []
    for line in code.splitlines():
        if line.strip().startswith("import "):
            continue
        lines.append(line)
    return "\n".join(lines)

def run_code_in_backend(code):
    """
    Runs code in the current Python kernel and returns any printed output.
    """
    # Capture stdout
    backup_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured

    try:
        # We use Python's built-in exec in a namespace 
        # so we can share dataframes but not pollute global namespace too much.
        # You can also use IPython.get_ipython().run_cell if desired.
        exec(code, globals(), globals())
    except Exception as e:
        print(f"[Error while running code]: {e}")
    finally:
        sys.stdout = backup_stdout

    return captured.getvalue()


###############################################################################
# 5. Notebook-building functions
###############################################################################
def append_code_cell(nb, code):
    """
    1) Remove duplicate imports, clean code
    2) Run the code in the backend
    3) Append a new code cell to the 'nb' notebook with the code + captured output
    """
    final_code = remove_duplicate_imports(clean_code(code))
    output_text = run_code_in_backend(final_code)

    code_cell = new_code_cell(source=final_code)
    if output_text.strip():
        # Create a 'stream' output entry with the captured text
        code_cell.outputs = [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": output_text
            }
        ]
    nb.cells.append(code_cell)

def append_markdown_cell(nb, markdown_text):
    """
    Append a Markdown cell (e.g., final summary).
    """
    md_cell = new_markdown_cell(source=markdown_text)
    nb.cells.append(md_cell)


###############################################################################
# 6. Main automation function for multi-DataFrame exploration
###############################################################################
def auto_explore_dataframes(dataframes, user_goal, output_notebook="auto_explore_results.ipynb"):
    """
    Automates multi-step data exploration with multiple DataFrames.
    
    Steps:
      1) Request a plan from LLM in JSON format.
      2) Parse it to get steps.
      3) For each step, request Python code from LLM and execute it.
      4) Check if more steps are needed.
      5) Generate a final summary in a Markdown cell.
      6) Save all code+outputs + final summary to a new .ipynb file.
    """

    # Create a new notebook structure
    nb = new_notebook()
    nb.cells = []

    # Summaries of each DataFrame for context
    df_summaries = []
    for name, df in dataframes.items():
        df_summaries.append(
            f"- **{name}**: columns={list(df.columns)};\n"
            f"  Sample:\n  {df.head(2).to_dict()}\n"
        )
    combined_df_info = "\n".join(df_summaries)

    #---------------------------------------------------------------------------
    # A) Prompt the LLM for a plan (JSON: { "steps": [...] })
    #---------------------------------------------------------------------------
    plan_prompt = f"""
We have multiple DataFrames loaded in memory:
{combined_df_info}

The user wants to accomplish:
{user_goal}

Requirements:
- Return only a JSON object of the form:
  {{
    "steps": [
       "Step 1 description",
       "Step 2 description"
    ]
  }}
- No extra commentary or Markdown. 
- Do not recreate or redefine DataFrames in any plan steps.
"""
    plan_json_str = call_llm_for_json_steps(plan_prompt, temperature=0.0)

    # We expect valid JSON that matches the "steps" schema; parse it:
    steps = []
    try:
        plan_data = json.loads(plan_json_str)
        steps = plan_data.get("steps", [])
    except json.JSONDecodeError:
        print("[Plan Parsing] Invalid JSON returned. Skipping steps.")

    #---------------------------------------------------------------------------
    # B) Execute each step by requesting code from the LLM and running it
    #---------------------------------------------------------------------------
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
        generated_code = call_llm_for_text(code_prompt)
        # Add code cell to the notebook
        append_code_cell(nb, generated_code)

        # (Optional) We can record the step + output for the final summary
        # Since append_code_cell runs the code, we can re-run the same logic:
        cleaned_code = remove_duplicate_imports(clean_code(generated_code))
        output_str = run_code_in_backend(cleaned_code)
        all_outputs.append((step, output_str))

    #---------------------------------------------------------------------------
    # C) Check if more steps are needed (follow-up)
    #---------------------------------------------------------------------------
    followup_prompt = f"""
We have executed the following steps:
{json.dumps(steps, indent=2)}

Do we need additional steps to achieve the goal: "{user_goal}"?
- If yes, provide more steps in the SAME JSON format:
  {{
    "steps": [
      "Additional step 1",
      "Additional step 2"
    ]
  }}
- Otherwise, say "No more steps needed."
"""
    followup_json_str = call_llm_for_json_steps(followup_prompt, temperature=0.0)
    followup_steps = []
    try:
        followup_data = json.loads(followup_json_str)
        followup_steps = followup_data.get("steps", [])
    except json.JSONDecodeError:
        # Possibly "No more steps needed." or invalid JSON
        pass

    # If more steps are suggested, you could optionally run them here.
    # For demonstration, we’ll just note them in the final summary.

    #---------------------------------------------------------------------------
    # D) Create a final summary as Markdown
    #---------------------------------------------------------------------------
    summary_prompt = f"""
Below is a list of steps executed and their corresponding outputs:
{all_outputs}

Please provide a concise summary of the key findings (plain text).
No JSON needed.
"""
    summary = call_llm_for_text(summary_prompt).strip()

    # Add final summary Markdown cell
    final_md_text = "## Final Conclusion & Findings\n\n" + summary
    if followup_steps:
        final_md_text += (
            "\n\n**Additional Steps Suggested by LLM**:\n" +
            "\n".join(f"- {s}" for s in followup_steps)
        )
    append_markdown_cell(nb, final_md_text)

    #---------------------------------------------------------------------------
    # E) Save the notebook to disk
    #---------------------------------------------------------------------------
    with open(output_notebook, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"[DONE] Notebook saved to: {output_notebook}")


###############################################################################
# 7. Example usage
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

    goal_example = "Explore the customers and their orders for some basic stats."

    # Run the auto-exploration
    auto_explore_dataframes(dataframes_dict, goal_example, "auto_explore_output.ipynb")
