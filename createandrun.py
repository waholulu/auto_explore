import io
import sys
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
    Creates a new Jupyter code cell with 'code' text and auto-runs it
    IF in a classic Jupyter Notebook environment.
    Otherwise, logs a message to the browser console.
    """
    escaped_code = code.replace("`", "\\`")
    js_script = f"""
    if (typeof Jupyter !== 'undefined' && Jupyter.notebook) {{
        // Classic Jupyter Notebook: we can insert and run a new cell
        var code = `{escaped_code}`;
        var cell = Jupyter.notebook.insert_cell_below('code');
        cell.set_text(code);
        Jupyter.notebook.select(cell);
        Jupyter.notebook.execute_cell_and_select_below();
    }} else {{
        // Fallback for JupyterLab / Colab / other environments
        console.log("Jupyter environment not found. Skipping cell creation.");
    }}
    """
    display(Javascript(js_script))

def create_and_run_cell(code):
    """
    Runs code in the backend, then attempts to insert a new visible cell 
    with the same code. Returns the captured output from the backend execution.
    """
    output = run_code_in_backend(code)
    create_code_cell_visually(code)
    return output

# Test usage
if __name__ == "__main__":
    test_code = "print('Hello from the test code cell!')"
    result = create_and_run_cell(test_code)
    print("Backend output captured:", result)
