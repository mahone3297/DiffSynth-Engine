import unittest
import torch

from diffsynth_engine import ZImagePipelineConfig
from diffsynth_engine.pipelines import ZImageOmniBasePipeline
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestZImagePipeline(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        config = ZImagePipelineConfig(
            model_path=fetch_model("Tongyi-MAI/Z-Image-Omni-Base", path="transformer/*.safetensors"),
            encoder_path=fetch_model("Tongyi-MAI/Z-Image-Omni-Base", path="text_encoder/*.safetensors"),
            vae_path=fetch_model("Tongyi-MAI/Z-Image-Omni-Base", path="vae/*.safetensors"),
            image_encoder_path=fetch_model("Tongyi-MAI/Z-Image-Omni-Base", path="siglip/model.safetensors"),
            model_dtype=torch.bfloat16,
            encoder_dtype=torch.bfloat16,
            vae_dtype=torch.float32,
            batch_cfg=True,
        )
        cls.pipe = ZImageOmniBasePipeline.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        del cls.pipe

    def test_txt2img(self):
        prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
        image = self.pipe(
            prompt=prompt,
            negative_prompt="",
            cfg_scale=5.0,
            width=1024,
            height=1024,
            num_inference_steps=40,
            seed=42,
        )
        self.assertImageEqualAndSaveFailed(image, "z_image/z_image_omni_base_t2i.png", threshold=0.95)

        prompt = "Change the women's clothes to white cheongsam, keep other content unchanged"
        image = self.pipe(
            prompt=prompt,
            negative_prompt="",
            edit_image=[image],
            height=1280,
            width=768,
            num_inference_steps=40,
            cfg_scale=4.0,
            seed=43,
        )
        self.assertImageEqualAndSaveFailed(image, "z_image/z_image_omni_base_i2i.png", threshold=0.95)


if __name__ == "__main__":
    unittest.main()
