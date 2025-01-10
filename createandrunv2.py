import io
import sys
import json
from IPython.display import Javascript, display
from IPython import get_ipython

def run_code_in_backend(code):
    """Runs code in the current Python kernel and returns any printed output."""
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
    Creates a new code cell in classic Jupyter Notebook, 
    sets its text, and executes it.
    """
    # Convert the Python code into a properly escaped JS string
    code_json = json.dumps(code)
    js_script = f"""
    if (typeof Jupyter !== 'undefined' && Jupyter.notebook) {{
        // Confirm environment
        console.log("Detected classic Jupyter Notebook environment.");
        
        // Insert a new cell, set text, and run it
        var new_cell = Jupyter.notebook.insert_cell_below('code');
        new_cell.set_text({code_json});
        Jupyter.notebook.select(new_cell);
        Jupyter.notebook.execute_cell_and_select_below();
    }} else {{
        console.log("Jupyter environment not found or not classic Notebook. Skipping cell creation.");
    }}
    """
    display(Javascript(js_script))

def create_and_run_cell(code):
    """Runs code in the Python backend, then inserts a new visible cell with the same code."""
    output = run_code_in_backend(code)
    create_code_cell_visually(code)
    return output

# Test usage
if __name__ == "__main__":
    result = create_and_run_cell("print('Hello from create_and_run_cell!')")
    print("Backend output captured:", repr(result))
