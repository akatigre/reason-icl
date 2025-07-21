from PIL import Image
from utils.vis_utils import parse_points, parse_bounding_boxes

def plan_wo_tags(question, answer=None):
    if answer is None:
        answer = "YOUR ANSWER"
    
    sys_content = "Think step by step and provide the final answer in \\boxed{}."
    user_content = question + f" Output the final answer as \\boxed{answer}."
    messages = [
        {
            "role": "system",
            "content": sys_content
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    return messages

class MyCoT():
    def __init__(self, **kwargs):
        self.multi_modal_tags = """
            <focus> for examining a specific region of the image, thinking where to focus to solve the question. </focus>
            <spatial> for reasoning about spatial relationships between parts of the image </spatial>
            """
        self.multi_modal_template = """
            <focus> ... </focus> OR Not Used
            <spatial> ... </spatial> OR Not Used
            """
            
    def plan_wo_tags(self, question, answer=None, image=None, **kwargs):
        if answer is None:
            answer = "YOUR ANSWER"
        
        if image is not None:
            sys_content = [
                    {
                        "type": "text", 
                        "text": "Think step by step and provide the final answer in \\boxed{}."
                    },
                ]
            user_content = [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": question + f" Output the final answer as \\boxed{answer}."
                }
            ]
        else:
            sys_content = "Think step by step and provide the final answer in \\boxed{}."
            user_content = question + f" Output the final answer as \\boxed{answer}."
        messages = [
            {
                "role": "system",
                "content": sys_content
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        return messages
                
           
    def plan_with_tags(self, content, answer=None, multi_modal=True):
        if answer is None:
            answer = "your answer"
        elif isinstance(answer, str):
            pass
        else:
            answer = answer[0]
        
        contents = [
            {
                "type": "text", 
                "text": f"""
                You are a reasoning assistant. Given a visual or multimodal question, your task is to:
                
                Step 1. Generate a step-by-step reasoning plan wrapped in <think> ... </think>
                Each step must be wrapped in a single tool tag defined below.
 
                <logic> for logical reasoning </logic>
                <commonsense> for commonsense reasoning. </commonsense>
                <external> for bringing in external knowledge or facts </external>
                {self.multi_modal_tags if multi_modal else ""}
                <conclude> for concluding the reasoning and points to {answer} </conclude>
                Always use <conclude> ... </conclude> at the end of the reasoning plan.
                
                Step 2. Provide the final answer in \\boxed.
                Keep it short and easy to verify, with no explanation or additional text. 
                
                
                Question:
                """
               
            }] + content + [{
                "type": "text",
                "text": f"""
                    Question: 
                    
                    [Output Format]:
                    List of steps: 
                    """
                    + 
                    r"Answer: \\boxed{answer}"
            }]

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text", 
                        "text": """
                        You are a helpful assistant that generates a reasoning plan to answer a question.
                        """
                    },
                ]
            },
            {
                "role": "user",
                "content": contents
            }
        ]
        return messages

    def _visualize_thinking(self, pil_image_path, thinking_step, detect_form, plot_fn):
        pil_image = Image.open(pil_image_path)
        
        if detect_form == "bbox":
            prompt = thinking_step + "\nLocate the objects in the image, output their bbox coordinates using JSON format, list of dicts. The key is 'bbox_2d'."
            parse_fn = parse_bounding_boxes
        elif detect_form == "points":
            prompt = thinking_step + "\nLocate the objects in the image, output their points using JSON format, list of dicts. The key is 'points'."
            parse_fn = parse_points
        else:
            raise ValueError(f"Unknown tool type: {detect_form}. Should be one of bbox or points")

        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "image": pil_image_path
                    }
                ]
            }
        ]
        
        # Process with the model
        inputs = self.process_fn(messages=messages)
        response = self.generate_fn(inputs=inputs, max_token=100)
        
        # Get image dimensions
        if "image_grid_thw" in inputs:
            input_height = inputs['image_grid_thw'][0][1] * 14
            input_width = inputs['image_grid_thw'][0][2] * 14
        else:
            input_height, input_width = inputs["pixel_values"].shape[-2:]
        
        # Visualize the results
        pil_image.thumbnail([640, 640], Image.Resampling.LANCZOS)
        bbox_or_point = parse_fn(pil_image, response[0], input_width, input_height)
        for idx, bp in enumerate(bbox_or_point):
            from utils.vis_utils import colors
            color = colors[idx % len(colors)]
            pil_image = plot_fn(pil_image, bp, color=color)
        return pil_image

