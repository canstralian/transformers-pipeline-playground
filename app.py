import gradio as gr
from transformers import pipeline

class HuggingfaceHubSearch(gr.inputs.Textbox):
    def __init__(self, label, placeholder, search_type, sumbit_on_select):
        super().__init__(label=label, placeholder=placeholder)
        self.search_type = search_type
        self.sumbit_on_select = sumbit_on_select

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
    def get_interface_from_repo(
        repo_id: str, progress: gr.Progress = gr.Progress()
    ):
        try:
            progress(0.5, desc="Loading model")
            pipe = pipeline(model=repo_id)
            progress(1.0, desc="Model loaded")
            return gr.Interface.from_pipeline(pipe, flagging_mode="never")
        except Exception as e:
            return gr.Markdown(
                f"This model is not supported. It might be too large or it does not work with Gradio. Try another model. Failed with expection: {e}"
            )


if __name__ == "__main__":
    demo.launch()

