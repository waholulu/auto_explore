import openai
import pandas as pd
import json
from IPython.display import Javascript, display, Markdown
from IPython import get_ipython
import io
import sys

###############################################################################
# 1. Setup: Configure your OpenAI API Key and LLM Client
###############################################################################

openai.api_key = "YOUR_OPENAI_API_KEY"

###############################################################################
# 2. LLM function using the OpenAI ChatCompletion API
###############################################################################

def call_llm(prompt, model="gpt-4", temperature=0.0):
    """
    Calls the OpenAI ChatCompletion endpoint with the given prompt.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are an AI assistant that provides helpful, "
                        "accurate Python code for data exploration tasks. "
                        "Return only valid JSON if requested."
                    )
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[LLM Error]: {e}")
        return ""

###############################################################################
# 3. Helper / Utility functions
###############################################################################

def extract_json_substring(response_text):
    """
    Extract the first JSON object found (from '{' to matching '}').
    """
    start_idx = response_text.find("{")
    end_idx = response_text.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return None
    return response_text[start_idx : end_idx + 1]

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
    """
    Creates and auto-runs a new Jupyter code cell with the given 'code' text.
    """
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
    """
    Cleans up, executes code in the backend, and creates a new cell in the UI.
    Returns the code's printed output.
    """
    final_code = remove_duplicate_imports(code)
    output = run_code_in_backend(final_code)
    create_code_cell_visually(final_code)
    return output

###############################################################################
# 4. Main automation function for multi-DataFrame exploration
###############################################################################

def auto_explore_dataframes(dataframes, user_goal, llm_function):
    """
    Automates multi-step data exploration with multiple DataFrames.
    
    Steps:
      1) Request a plan from LLM in JSON format.
      2) Parse it to get steps.
      3) For each step, request Python code from LLM and execute it.
      4) Check if more steps are needed.
      5) Generate a final summary Markdown cell with findings.
    """
    # Summaries of each DataFrame for context
    df_summaries = []
    for name, df in dataframes.items():
        df_summaries.append(
            f"- **{name}**: columns={list(df.columns)};\n"
            f"  Sample:\n  {df.head(2).to_dict()}\n"
        )
    combined_df_info = "\n".join(df_summaries)

    #---------------------------------------------------------------------------
    # A) Prompt LLM for a plan (JSON: { "steps": [...] })
    #---------------------------------------------------------------------------
    plan_prompt = f"""
We have multiple DataFrames loaded in memory:
{combined_df_info}

The user wants to accomplish:
{user_goal}

Requirements:
- Return only a JSON object, e.g.:
  {{
    "steps": [
       "Step 1 description",
       "Step 2 description"
    ]
  }}
- No extra commentary or Markdown. 
- Do not recreate or redefine DataFrames in any plan steps.
"""
    plan_json_str = llm_function(plan_prompt)
    extracted_plan = extract_json_substring(plan_json_str)

    # If LLM fails to produce valid JSON, handle gracefully
    steps = []
    if extracted_plan:
        try:
            plan_data = json.loads(extracted_plan)
            steps = plan_data.get("steps", [])
        except json.JSONDecodeError:
            print("[Plan Parsing] Invalid JSON, skipping steps.")

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
        generated_code = llm_function(code_prompt)
        generated_code = clean_code(generated_code)
        output_str = create_and_run_cell(generated_code)
        all_outputs.append((step, output_str))

    #---------------------------------------------------------------------------
    # C) Check if more steps are needed
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
    followup_response = llm_function(followup_prompt)
    extracted_followup = extract_json_substring(followup_response)
    if extracted_followup:
        try:
            followup_data = json.loads(extracted_followup)
            # If you want to auto-run additional steps, repeat the same loop here
        except json.JSONDecodeError:
            pass

    #---------------------------------------------------------------------------
    # D) Create a final Markdown cell summarizing the exploration
    #---------------------------------------------------------------------------
    summary_prompt = f"""
Below is a list of steps executed and their corresponding outputs:
{all_outputs}

Please provide a concise summary of the key findings (plain text).
No JSON needed.
"""
    summary = llm_function(summary_prompt).strip()
    display(Markdown(f"## Final Conclusion & Findings\n\n{summary}"))

###############################################################################
# 5. Example usage
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

    # Bundle them for the automation
    dataframes_dict = {
        "df_customers": df_customers,
        "df_orders": df_orders
    }

    goal_example = "Explore the customers and their orders for some basic stats."

    # Run the auto-exploration
    auto_explore_dataframes(dataframes_dict, goal_example, call_llm)
