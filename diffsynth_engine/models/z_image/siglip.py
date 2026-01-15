from transformers import Siglip2VisionModel, Siglip2VisionConfig, Siglip2ImageProcessorFast
import torch


class Siglip2ImageEncoder(Siglip2VisionModel):
    def __init__(self, **kwargs):
        config = Siglip2VisionConfig(
            attention_dropout = 0.0,
            dtype = "bfloat16",
            hidden_act = "gelu_pytorch_tanh",
            hidden_size = 1152,
            intermediate_size = 4304,
            layer_norm_eps = 1e-06,
            model_type = "siglip2_vision_model",
            num_attention_heads = 16,
            num_channels = 3,
            num_hidden_layers = 27,
            num_patches = 256,
            patch_size = 16,
            transformers_version = "4.57.1"
        )
        super().__init__(config)
        self.processor = Siglip2ImageProcessorFast(
            **{
                "data_format": "channels_first",
                "default_to_square": True,
                "device": None,
                "disable_grouping": None,
                "do_convert_rgb": None,
                "do_normalize": True,
                "do_pad": None,
                "do_rescale": True,
                "do_resize": True,
                "image_mean": [
                    0.5,
                    0.5,
                    0.5
                ],
                "image_processor_type": "Siglip2ImageProcessorFast",
                "image_std": [
                    0.5,
                    0.5,
                    0.5
                ],
                "input_data_format": None,
                "max_num_patches": 256,
                "pad_size": None,
                "patch_size": 16,
                "processor_class": "Siglip2Processor",
                "resample": 2,
                "rescale_factor": 0.00392156862745098,
                "return_tensors": None,
            }
        )
        
    def forward(self, image, torch_dtype=torch.bfloat16, device="cuda"):
        siglip_inputs = self.processor(images=[image], return_tensors="pt").to(device)
        shape = siglip_inputs.spatial_shapes[0]
        hidden_state = super().forward(**siglip_inputs).last_hidden_state
        B, N, C = hidden_state.shape
        hidden_state = hidden_state[:, : shape[0] * shape[1]]
        hidden_state = hidden_state.view(shape[0], shape[1], C)
        hidden_state = hidden_state.to(torch_dtype)
        return hidden_state

    @classmethod
    def from_state_dict(cls, state_dict, device: str, dtype: torch.dtype):
        model = cls()
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model
