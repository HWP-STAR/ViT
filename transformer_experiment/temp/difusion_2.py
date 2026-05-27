import torch
from diffusers import StableDiffusionXLPipeline

cache_dir = "../../../data"
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# 直接在 from_pretrained 中开启 4bit 量化
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,       # 基础计算精度设为半精度
    use_safetensors=True,            # 使用安全的权重格式
    variant="fp16",                  # 加载 fp16 变体版本
    cache_dir=cache_dir,
    
    # 👇 核心修改：diffusers 库自带的 4bit 量化参数
    load_in_4bit=True,               # 开启 4bit 量化加载
    bnb_4bit_compute_dtype=torch.float16, # 指定计算时的数据类型
    bnb_4bit_use_double_quant=True,  # 开启双重量化，进一步压缩显存
    bnb_4bit_quant_type="nf4"        # 使用 NF4 量化类型
)

# 建议配合 CPU offload 进一步降低峰值显存占用
pipe.enable_model_cpu_offload()

#pipe.to("cuda")

print('=='*30,'download ok','=='*30)
'''
# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

image = pipe(prompt=prompt).images[0]
# 保存图片到本地
image.save("output.png")
print("图片生成完毕，已保存为 output.png")
'''
