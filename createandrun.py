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
    Creates a new Jupyter code cell with 'code' text, and auto-runs it.
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
    Runs code in the backend, then inserts a new visible cell with the same code.
    Returns the captured output from the backend execution.
    """
    output = run_code_in_backend(code)
    create_code_cell_visually(code)
    return output

# Test usage
if __name__ == "__main__":
    result = create_and_run_cell("print('Hello from the test code cell!')")
    print("Backend output captured:", result)
