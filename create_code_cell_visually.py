import json
from IPython.display import Javascript, display

def create_code_cell_visually(code):
    code_json = json.dumps(code)  # escapes special chars safely
    js_script = f"""
    if (typeof Jupyter !== 'undefined' && Jupyter.notebook) {{
        console.log("Detected classic Jupyter Notebook environment.");
        var new_cell = Jupyter.notebook.insert_cell_below('code');
        new_cell.set_text({code_json});
        Jupyter.notebook.select(new_cell);
        Jupyter.notebook.execute_cell_and_select_below();
    }} else {{
        console.log("Jupyter environment not found or not classic Notebook. Skipping cell creation.");
    }}
    """
    display(Javascript(js_script))

create_code_cell_visually("print('Hello from the new cell!')")
