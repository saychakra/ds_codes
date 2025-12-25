"""
DS Code Playground - Interactive exploration and execution environment
"""

import streamlit as st
from utils.code_loader import (
    flatten_structure,
    get_code_structure,
    get_summary_from_docstring,
    load_description,
    read_file_content,
)

st.set_page_config(
    page_title="DS Code Playground", page_icon="🧪", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
    }
    .code-container {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧪 DS Code Playground")
st.markdown("_Interactive exploration, understanding, and execution of your ML/DS codes_")

# Initialize session state
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "code_structure" not in st.session_state:
    st.session_state.code_structure = get_code_structure()

# Sidebar navigation
with st.sidebar:
    st.header("📂 Navigation")

    # Get all files as flat list
    all_files = flatten_structure(st.session_state.code_structure)
    file_names = [name for name, _ in all_files]
    file_dict = {name: info for name, info in all_files}

    # File selector
    selected = st.selectbox(
        "Select a file to explore:",
        file_names,
        key="file_selector",
        help="Choose a Python script or Jupyter notebook",
    )

    if selected:
        st.session_state.selected_file = file_dict[selected]

    st.divider()

    # Statistics
    st.subheader("📊 Workspace Stats")
    total_py = len([f for f in all_files if f[0].endswith(".py")])
    total_nb = len([f for f in all_files if f[0].endswith(".ipynb")])
    col1, col2 = st.columns(2)
    col1.metric("Python Files", total_py)
    col2.metric("Notebooks", total_nb)

# Main content area
if st.session_state.selected_file:
    file_info = st.session_state.selected_file
    file_path = file_info["path"]
    file_name = file_info["name"]

    # Header with file info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(f"📄 {file_name}")
    with col2:
        st.caption(f"Type: {file_info['type'].upper()}")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Overview", "💻 Code", "🎮 Playground", "📝 Description"])

    with tab1:
        st.subheader("File Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Type", file_info["type"])
        with col2:
            st.metric("Path", file_path.split("/")[-1])

        # Show docstring summary if available
        if file_info["type"] == "script":
            content = read_file_content(file_path)
            docstring = get_summary_from_docstring(content)
            if docstring:
                st.info(f"**Summary:** {docstring}")

    with tab2:
        st.subheader("Source Code")
        content = read_file_content(file_path)

        if file_info["type"] == "notebook":
            st.info(
                "📓 Notebook files cannot be directly edited here. View in Jupyter or convert to Python."
            )
            with st.expander("Show raw notebook JSON"):
                st.code(content[:1000] + "...", language="json")
        else:
            # Show code with syntax highlighting
            st.code(content, language="python", line_numbers=True)

            # Download button
            st.download_button(
                label="⬇️ Download Code", data=content, file_name=file_name, mime="text/plain"
            )

    with tab3:
        st.subheader("🎮 Interactive Playground")

        if file_info["type"] == "notebook":
            st.warning("⚠️ Notebook execution support coming soon!")
        else:
            st.info("✨ Execute code and experiment with parameters")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.text("Key parameters and functions detected in this file:")
                content = read_file_content(file_path)

                # Extract imports
                imports = [
                    line
                    for line in content.split("\n")
                    if line.strip().startswith("import ") or line.strip().startswith("from ")
                ][:10]
                if imports:
                    st.caption("Imports:")
                    for imp in imports:
                        st.code(imp, language="python")

            with col2:
                st.info("""
                **How to use:**
                1. Modify parameters in the sidebar
                2. Click 'Run Code'
                3. See results below
                """)

            # Code execution area
            st.divider()

            # Execution options
            col1, col2, col3 = st.columns(3)
            with col1:
                run_code = st.button("▶️ Run Code", use_container_width=True, type="primary")
            with col2:
                capture_output = st.checkbox("Capture output", value=True)
            with col3:
                show_warnings = st.checkbox("Show warnings", value=False)

            if run_code:
                st.subheader("📊 Execution Results")

                try:
                    import sys
                    import warnings
                    from io import StringIO

                    content = read_file_content(file_path)

                    # Setup execution environment
                    execution_namespace = {
                        "__name__": "__main__",
                        "__file__": file_path,
                    }

                    # Capture output if enabled
                    if capture_output:
                        old_stdout = sys.stdout
                        old_stderr = sys.stderr
                        sys.stdout = StringIO()
                        sys.stderr = StringIO()

                    # Handle warnings
                    if not show_warnings:
                        warnings.filterwarnings("ignore")

                    # Execute code
                    exec(content, execution_namespace)

                    # Get output
                    if capture_output:
                        output = sys.stdout.getvalue()
                        errors = sys.stderr.getvalue()
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr

                        if output:
                            st.success("✅ Code executed successfully!")
                            st.text_area("Output:", output, height=200, disabled=True)
                        else:
                            st.success("✅ Code executed successfully! (No output)")

                        if errors:
                            st.warning("⚠️ Warnings/Errors:")
                            st.text_area("Messages:", errors, height=100, disabled=True)
                    else:
                        st.success("✅ Code executed successfully!")

                except Exception as e:
                    st.error(f"❌ Execution Error: {type(e).__name__}")
                    st.code(str(e), language="python")

                    # Show traceback
                    import traceback

                    with st.expander("📋 Full Traceback"):
                        st.code(traceback.format_exc())

    with tab4:
        st.subheader("📝 Documentation")
        description = load_description(file_name)

        if description:
            st.markdown(description)
        else:
            st.info(f"""
            No description file found for this code.

            To add a description, create a markdown file at:
            `playground/descriptions/{file_name.replace(".py", ".md").replace(".ipynb", ".md")}`
            """)

            # Template
            with st.expander("📋 Description Template"):
                template = f"""# {file_name}

## What does this code do?
Brief explanation of the purpose and functionality.

## Key Components
- **Component 1**: Description
- **Component 2**: Description

## Learning Concepts
- Concept 1
- Concept 2

## Usage Example
```python
# Example usage here
```

## Related Files
- Link to related code files

## Further Reading
- Relevant documentation or resources
"""
                st.code(template, language="markdown")

else:
    st.info("👈 Select a file from the sidebar to get started!")

    st.markdown("""
    ## 🚀 Getting Started

    This playground helps you:
    - **Browse** all your ML/DS code files
    - **Understand** what each file does with descriptions
    - **Explore** source code with syntax highlighting
    - **Experiment** with code execution and parameters
    - **Learn** by reviewing and running code samples

    ### How to add descriptions:
    1. Go to `playground/descriptions/`
    2. Create a markdown file matching your code filename
    3. Write clear explanations of what the code does
    4. Refresh the app to see your description
    """)
