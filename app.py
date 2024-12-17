import gradio as gr
from gradio_huggingfacehub_search import HuggingfaceHubSearch
from transformers import pipeline

with gr.Blocks() as demo:
    gr.Markdown("## 🐇 Transformers Pipeline Playground")
    gr.Markdown(
        "Search for a model on the Hub en explore its output performance on CPU. Some interesting categories are [Text Classification](https://huggingface.co/models?pipeline_tag=image-classification&sort=trending), [Token Classification](https://huggingface.co/models?pipeline_tag=token-classification&sort=trending), [Question Answering](https://huggingface.co/models?pipeline_tag=question-answering&sort=trending) or [Image Classification](https://huggingface.co/models?pipeline_tag=image-classification&sort=trending)."
    )
    search_in = HuggingfaceHubSearch(
        label="Hub Search",
        placeholder="Search for a model",
        search_type="model",
        sumbit_on_select=True,
    )

    @gr.render(inputs=[search_in], triggers=[search_in.submit])
    def get_interface_from_repo(repo_id: str, progress: gr.Progress = gr.Progress()):
        progress(0.0, desc="Loading model")
        pipe = pipeline(model=repo_id)
        progress(1.0, desc="Model loaded")
        gr.Interface.from_pipeline(pipe)


if __name__ == "__main__":
    demo.launch()
