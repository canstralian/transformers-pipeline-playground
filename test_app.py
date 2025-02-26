import unittest
from unittest.mock import patch
import gradio as gr
from gradio_huggingfacehub_search import HuggingfaceHubSearch
from transformers import pipeline


class TestGradioApp(unittest.TestCase):
    @patch('transformers.pipeline')
    def test_get_interface_from_repo_success(self, mock_pipeline):
        # Mocking a successful pipeline loading
        mock_pipe = "mocked_pipeline"  # Replace this with your expected mocked pipeline output
        mock_pipeline.return_value = mock_pipe

        # Simulating Gradio interface
        with gr.Blocks() as demo:
            search_in = HuggingfaceHubSearch(
                label="Hub Search",
                placeholder="Search for a model",
                search_type="model",
                submit_on_select=True,
            )

            @gr.render(inputs=[search_in], triggers=[search_in.submit])
            def get_interface_from_repo(repo_id: str, progress: gr.Progress = gr.Progress()):
                pipe = pipeline(model=repo_id)
                return gr.Interface.from_pipeline(pipe, flagging_mode="never")

            # Trigger the function with a mock repo_id
            result = get_interface_from_repo("test_model_id")
            self.assertIsNotNone(result)
            self.assertEqual(result, gr.Interface.from_pipeline(mock_pipe, flagging_mode="never"))

    @patch('transformers.pipeline')
    def test_get_interface_from_repo_failure(self, mock_pipeline):
        # Mocking a failure in pipeline loading
        mock_pipeline.side_effect = Exception("Model loading failed")

        # Simulating Gradio interface
        with gr.Blocks() as demo:
            search_in = HuggingfaceHubSearch(
                label="Hub Search",
                placeholder="Search for a model",
                search_type="model",
                submit_on_select=True,
            )

            @gr.render(inputs=[search_in], triggers=[search_in.submit])
            def get_interface_from_repo(repo_id: str, progress: gr.Progress = gr.Progress()):
                try:
                    pipe = pipeline(model=repo_id)
                    return gr.Interface.from_pipeline(pipe, flagging_mode="never")
                except Exception as e:
                    return gr.Markdown(f"This model is not supported. Failed with exception: {e}")

            # Trigger the function with a mock repo_id
            result = get_interface_from_repo("test_model_id")
            self.assertIsInstance(result, gr.Markdown)
            self.assertIn("This model is not supported", result.value)

if __name__ == "__main__":
    unittest.main()
