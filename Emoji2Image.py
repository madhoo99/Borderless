from PIL import Image, ImageDraw, ImageFont

# Load the font file
font_file = 'seguiemj.ttf'
font_size = 150
font = ImageFont.truetype(font_file, font_size)

# Create a new RGBA image with a transparent background
width, height = 200, 200
image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

# Draw the emoji using the font
text = '😀❤☃'
# color = (255, 0, 0) # Red color
# text_width, text_height = font.getsize(text)
# draw = ImageDraw.Draw(image)
# draw.text(((width - text_width) / 2, (height - text_height) / 2), text, font=font)

# Get the bounding box of the text
bbox = font.getbbox(text)

# Create a new image with dimensions based on the bounding box
image = Image.new("RGBA", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255, 0))

# Draw the text on the image
draw = ImageDraw.Draw(image)
draw.text((0, -bbox[1]), text, font=font, embedded_color=True)

# Save the image to disk
image.save('colored_emoji.png')

