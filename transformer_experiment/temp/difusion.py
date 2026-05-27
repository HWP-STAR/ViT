from diffusers import DiffusionPipeline
import torch

cache_dir="../../../data"
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
                                         cache_dir=cache_dir)
pipe.to("cuda")

print('=='*30,'download ok','=='*30)

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

image = pipe(prompt=prompt).images[0]
# 保存图片到本地
image.save("output.png")
print("图片生成完毕，已保存为 output.png")

