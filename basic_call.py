from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

processor = Pix2StructProcessor.from_pretrained("google/matcha-chart2text-pew")
model = Pix2StructForConditionalGeneration.from_pretrained(
    "google/matcha-chart2text-pew"
)

url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/20294671002019.png"
image = Image.open(requests.get(url, stream=True).raw)
header_text = "Generate underlying data table of the figure below:"

inputs = processor(images=image, text=header_text, return_tensors="pt")
predictions = model.generate(**inputs, max_new_tokens=512)
print("model done")
print(processor.decode(predictions[0], skip_special_tokens=True))
print("decode done")
