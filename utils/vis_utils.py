import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm

import ast
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET
import math

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]
colors = [
        'red',
        'green',
        'blue',
        'yellow',
        'orange',
        'pink',
        'purple',
        'brown',
        'gray',
        'beige',
        'turquoise',
        'cyan',
        'magenta',
        'lime',
        'navy',
        'maroon',
        'teal',
        'olive',
        'coral',
        'lavender',
        'violet',
        'gold',
        'silver',
    ] + additional_colors

def decode_xml_points(text):
    try:
        root = ET.fromstring(text)
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            x = root.attrib.get(f'x{i+1}')
            y = root.attrib.get(f'y{i+1}')
            points.append([x, y])
        alt = root.attrib.get('alt')
        phrase = root.text.strip() if root.text else None
        return {
            "points": points,
            "alt": alt,
            "phrase": phrase
        }
    except Exception as e:
        print(e)
        return None

def parse_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """
    # Load the image
    font_name = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    img = im
    width, height = img.size
    draw_bbox = ImageDraw.Draw(img)

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    font = ImageFont.truetype(font_name, size=14)

    try:
        json_output = ast.literal_eval(bounding_boxes) # should be a list of dicts
        if not isinstance(json_output, list):
            json_output = [json_output]
        assert isinstance(json_output[0], dict) and "bbox_2d" in json_output[0], "Bounding boxes must be a list of dicts with 'bbox_2d' key"
    except Exception as e:
        end_idx = bounding_boxes.rfind('"}') + len('"}')
        truncated_text = bounding_boxes[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)
    bboxes = []
    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["bbox_2d"][1] / input_height * height)
        abs_x1 = int(bounding_box["bbox_2d"][0] / input_width * width)
        abs_y2 = int(bounding_box["bbox_2d"][3] / input_height * height)
        abs_x2 = int(bounding_box["bbox_2d"][2] / input_width * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        bboxes.append((abs_x1, abs_y1, abs_x2, abs_y2))
    return bboxes
        

def plot_connecting_arrows(img, bounding_boxes):
    draw = ImageDraw.Draw(img)
    centers = []
    for i, bounding_box in enumerate(bounding_boxes):
        # Calculate center point
        abs_x1, abs_y1, abs_x2, abs_y2 = bounding_box
        center_x = (abs_x1 + abs_x2) // 2
        center_y = (abs_y1 + abs_y2) // 2
        centers.append((center_x, center_y))

    start_x, start_y = centers[0]
    end_x, end_y = centers[1]
    
    # Draw the arrow line
    draw.line([(start_x, start_y), (end_x, end_y)], fill='red', width=3)
        
    # Draw arrow head
    arrow_length = 15
    angle = math.atan2(end_y - start_y, end_x - start_x)
    arrow_angle = math.pi / 6  # 30 degrees
    
    # Calculate arrow head points
    x1 = end_x - arrow_length * math.cos(angle - arrow_angle)
    y1 = end_y - arrow_length * math.sin(angle - arrow_angle)
    x2 = end_x - arrow_length * math.cos(angle + arrow_angle)
    y2 = end_y - arrow_length * math.sin(angle + arrow_angle)
    
    # Draw arrow head
    draw.line([(end_x, end_y), (x1, y1)], fill='red', width=3)
    draw.line([(end_x, end_y), (x2, y2)], fill='red', width=3)

    return img

def plot_bounding_boxes(img, bounding_box, color=None, label=None):
    draw = ImageDraw.Draw(img)
    if color is None:
        color = colors[0]
    abs_x1, abs_y1, abs_x2, abs_y2 = bounding_box
    draw.rectangle([(abs_x1, abs_y1), (abs_x2, abs_y2)], outline=color, width=3)
    if label:
        draw.text((abs_x1 + 8, abs_y1 + 6), label, fill=color)
    return img

def plot_cropping(img, bounding_box, color=None):
    if color is None:
        color = colors[0]
    abs_x1, abs_y1, abs_x2, abs_y2 = bounding_box
    img = img.crop((abs_x1, abs_y1, abs_x2, abs_y2))
    img.resize((640, 640), Image.Resampling.LANCZOS)
    return img
   
def plot_circle(img, point, color=None):
    draw = ImageDraw.Draw(img)
    abs_x, abs_y = point
    radius = 2
    color = colors[0]
    draw.ellipse([(abs_x-radius, abs_y-radius), (abs_x+radius, abs_y+radius)], outline=color, width=3)
    return img
    
def parse_points(im, text, input_width, input_height):
    img = im
    width, height = img.size
    xml_text = text.replace('```xml', '')
    xml_text = xml_text.replace('```', '')
    data = decode_xml_points(xml_text)
    if data is None:
        return None
    points = data['points']
    all_points = []
    for i, point in enumerate(points):
        abs_x1 = int(point[0])/input_width * width
        abs_y1 = int(point[1])/input_height * height
        all_points.append((abs_x1, abs_y1))
    return all_points

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output
