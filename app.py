import gradio as gr
from gradio_huggingfacehub_search import HuggingfaceHubSearch
from transformers import pipeline, Pipeline
from transformers.pipelines import PipelineException
from huggingface_hub.utils import ModelNotFoundError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Hugging Face Hub search component
search_in = HuggingfaceHubSearch(api_key="hf_YourAPITokenHere", submit_on_select=True)

# Function to load the selected model and create a pipeline
def load_model(model_id):
    try:
        logger.info(f"Loading model: {model_id}")
        model_pipeline = pipeline(model=model_id)
        logger.info("Model loaded successfully.")
        return model_pipeline
    except ModelNotFoundError:
        logger.error(f"Model '{model_id}' not found.")
        return None
    except PipelineException as e:
        logger.error(f"Error creating pipeline: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

# Function to process input data using the loaded pipeline
def process_input(model_pipeline, input_data):
    try:
        logger.info("Processing input data.")
        output = model_pipeline(input_data)
        logger.info("Processing complete.")
        return output
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return None

# Gradio interface setup
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Transformers Pipeline Playground")
        model_id = gr.Textbox(label="Enter Model ID from Hugging Face Hub")
        input_data = gr.Textbox(label="Input Data")
        output_data = gr.Textbox(label="Output Data")
        load_button = gr.Button("Load Model")
        process_button = gr.Button("Process Input")

        # Load model on button click
        def on_load_click():
            model_pipeline = load_model(model_id.value)
            if model_pipeline:
                process_button.click(
                    fn=lambda: process_input(model_pipeline, input_data.value),
                    inputs=[],
                    outputs=output_data,
                )
            else:
                output_data.value = "Failed to load model."

        load_button.click(on_load_click, inputs=[], outputs=[])

    return demo

# Run the Gradio interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()