# 导入StableDiffusionPipeline管道类，用于生成图像
from diffusers import StableDiffusionPipeline

# 加载预训练的Stable Diffusion模型权重，并将模型移动到CUDA设备上
# 这是为了加速图像生成过程，利用GPU的并行计算能力
pipe = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1").to("cuda")

# 定义生成图像的文本提示，结合了古诗词和油画风格的描述
# 这将指导模型生成具有飞流直下三千尺景象的油画风格图像
prompt = '飞流直下三千尺，油画'

# 使用管道生成图像，采用更高的guidance_scale以获得更贴近文本提示的图像结果
# guidance_scale的值为7.5，平衡了图像多样性和文本一致性
image = pipe(prompt, guidance_scale=7.5).images[0]

# 将生成的图像保存为文件，命名为“飞流.png”
# 这样可以持久化存储生成的图像，以便后续查看或使用
image.save("飞流.png")
