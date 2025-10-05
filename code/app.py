import gradio as gr
from llm_agent import split_profile, evaluate_profile, generate_resume


MAX_EXP = 100  # Maximum 100 experiences in this app

def read_file_content(file_path: str) -> str:
    """Read content from txt file."""
    if not file_path:
        return ""
    try:
        # Get file extension
        _, ext = os.path.splitext(file_path.lower())
        if ext == '.txt':
            # Plain text files
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Try to read as plain text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def process_file(file, resume_state):
    """Process uploaded file and return paragraph textboxes."""
    if file is None:
        return [gr.update(visible=False)] * MAX_EXP  # Hide all textboxes
    
    # Read file content
    content = read_file_content(file.name)
    
    if content.startswith("Error:"):
        # Return error in first textbox, hide others
        updates = [gr.update(value=content, visible=True, interactive=False)]
        updates.extend([gr.update(visible=False)] * (MAX_EXP-1))
        return updates
    
    # Parse into paragraphs
    paragraphs = split_profile(content)
    resume_state += "/n/n".join(paragraphs)
    
    # Create updates for textboxes
    updates = []
    for i in range(MAX_EXP):
        if i < len(paragraphs):
            updates.append(gr.update(
                value=paragraphs[i],
                visible=True,
                interactive=True,
                label=f"Paragraph {i+1}"
            ))
        else:
            updates.append(gr.update(visible=False))
    
    return updates + [resume_state]


def create_file_parser_interface(resume_state):
    """Create the file parser interface as a standalone component."""
    
    gr.Markdown("# File Upload and Paragraph Parser")
    gr.Markdown("Upload a text file, and it will be parsed into editable paragraphs.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload File",
                file_types=[".txt",],
                type="filepath"
            )
            
        with gr.Column(scale=2):
            gr.Markdown("## Parsed Resume")
            gr.Markdown("Edit the paragraphs below as needed:")
            
            # Create 10 textboxes for paragraphs (hidden initially)
            paragraph_textboxes = []
            for i in range(MAX_EXP):
                textbox = gr.Textbox(
                    label=f"Paragraph {i+1}",
                    lines=4,
                    max_lines=8,
                    visible=False,
                    interactive=True,
                    placeholder="Paragraph content will appear here..."
                )
                paragraph_textboxes.append(textbox)
    
    # Event handlers
    file_upload.change(
        fn=process_file,
        inputs=[file_upload, resume_state],
        outputs=paragraph_textboxes + [resume_state]
    )
    return resume_state


def create_resume_processor_interface(resume_state):
    
    gr.Markdown("# Job Evaluation & Resume Generation")
    
    with gr.Row():
        # Left column - Results
        with gr.Column(scale=1):
            gr.Markdown("### Processing Results")
            result_textbox = gr.Textbox(
                label="Results",
                lines=25,
                max_lines=30,
                placeholder="Results will appear here after you click Evaluate or Generate...",
                interactive=False
            )
            
            # Action buttons
            with gr.Row():
                evaluate_btn = gr.Button("📊 Evaluate", variant="primary", size="lg")
                generate_btn = gr.Button("✨ Generate", variant="secondary", size="lg")
            
            # Clear button
            clear_btn = gr.Button("🗑️ Clear All", size="sm")
        
        # Right column - Input
        with gr.Column(scale=1):
            gr.Markdown("### Paste Job Description")
            input_textbox = gr.Textbox(
                label="Paste Here",
                lines=25,
                max_lines=30,
                placeholder="Paste the job description here...\n\nThen click 'Evaluate' to see how fit you are or 'Generate' to create a taylored resume for the job.",
                interactive=True
            )
            
            # # Info box
            # with gr.Accordion("ℹ️ How to use", open=False):
            #     gr.Markdown("""
            #     ### Evaluate Mode:
            #     - Analyzes your text
            #     - Provides statistics (word count, sentence count, etc.)
            #     - Calculates readability metrics
            #     - Shows vocabulary richness
                
            #     ### Generate Mode:
            #     - Creates a summary
            #     - Finds most frequent words
            #     - Generates text variations
            #     - Creates alternative titles
            #     - Shows acrostic patterns
                
            #     **Tip**: Try pasting an article, essay, or any substantial text for best results!
            #     """)
    
    # Event handlers
    evaluate_btn.click(
        fn=evaluate_profile,
        inputs=[input_textbox, resume_state],
        outputs=[result_textbox]
    )
    
    generate_btn.click(
        fn=generate_resume,
        inputs=[input_textbox, resume_state],
        outputs=[result_textbox]
    )
    
    def clear_all_fields():
        return "", ""
    
    clear_btn.click(
        fn=clear_all_fields,
        outputs=[result_textbox, input_textbox]
    )


def create_blocks_based_app():
    """Create app using Blocks for maximum customization."""
    
    with gr.Blocks(title="Resume.AI", theme=gr.themes.Default()) as app:
        
        gr.Markdown("# Resume.AI")
        resume_state = gr.State("")
        
        with gr.Tabs():
            with gr.Tab("File Parser"):
                resume_state = create_file_parser_interface(resume_state)
            with gr.Tab("Evaluate/Generate"):
                create_resume_processor_interface(resume_state)
            # with gr.Tab("Settings"):
            #     gr.Markdown("## Application Settings")
            #     theme_choice = gr.Radio(
            #         ["Default", "Soft", "Monochrome"],
            #         label="Theme",
            #         value="Soft"
            #     )
            #     auto_save = gr.Checkbox(label="Auto-save results", value=True)
            #     save_settings_btn = gr.Button("Save Settings")
            #     settings_output = gr.Textbox(label="Settings Status")
                
            #     def save_app_settings(theme, auto_save_val, max_size):
            #         return f"Settings saved: Theme={theme}, Auto-save={auto_save_val}, Max file size={max_size}MB"
                
            #     save_settings_btn.click(
            #         save_app_settings,
            #         inputs=[theme_choice, auto_save, max_file_size],
            #         outputs=settings_output
            #     )
    return app


app = create_blocks_based_app()
app.launch(share=False, debug=True)
