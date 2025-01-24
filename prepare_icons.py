from PIL import Image

base_image = Image.open("base_icon.png")

sizes = [64, 48, 32, 16]
images = [base_image.resize((size, size)) for size in sizes]

base_image.save("main.ico", format="ICO", append_images=images)
