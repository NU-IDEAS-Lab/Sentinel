import base64
from io import BytesIO

from PIL import Image

from .llm_step import LLM_StepAgent
from models.prompts import SYS_PROMPT_STEP


class VLM_StepAgent(LLM_StepAgent):
    """
    Stepwise VLM agent that uses text + optional image context.
    First generate subgoals based on full scene info; later steps use image-only context to generate action.
    """

    def __init__(self, args):
        super().__init__(args)

    def get_next_action(self, task_desc, subgoals=None, image=None, action_history=None):
        user_prompt = self.create_prompt(task_desc, subgoals, action_history, image_available=image is not None)

        self.log("=" * 50)
        self.log("STEPWISE VLM ACTION PROMPT:")
        self.log(user_prompt)
        self.log("=" * 50)

        response_text = self.query_vlm(SYS_PROMPT_STEP, user_prompt, image=image)
        next_action = self.parse_single_action_response(response_text)

        self.conversation_history.append({
            'prompt': user_prompt,
            'response': response_text,
            'action': next_action,
        })

        self.log(f"Generated next action: {next_action}")
        return next_action

    def create_prompt(self, task_desc, subgoals, action_history=None, image_available=False):
        # Parse subgoals with improved handling for markdown format
        subgoals = self.extract_subgoals(subgoals)
        prompt = f"""
## Task: {task_desc}
## Subgoals to Achieve (in order):
"""
        
        # Add numbered subgoals for clarity
        if subgoals:
            for i, subgoal in enumerate(subgoals, 1):
                prompt += f"{i}. {subgoal}\n"
        else:
            prompt += "No valid subgoals found - will generate basic plan\n"
        prompt += f"""
## Current Observation:
An image of the current scene is provided. No object list is available.

"""

        prompt += "## Actions Taken So Far:\n"
        if action_history:
            for i, action in enumerate(action_history[-10:], 1):
                status = "✓" if action.get('success', True) else "✗"
                action_name = action.get('action', 'Unknown')
                prompt += f"{i}. {status} {action_name}"
                if action_name == "GetObjectInFrame":
                    x = action.get('x')
                    y = action.get('y')
                    result_obj = action.get('result_object_id')
                    prompt += f" x={x}, y={y} -> {result_obj}"
                elif 'object_id' in action and action.get('object_id'):
                    prompt += f" {action['object_id']}"
                if not action.get('success', True) and 'error' in action:
                    prompt += f" (Error: {action['error']})"
                prompt += "\n"
        else:
            prompt += "None yet - this is the first action.\n"

        prompt += """
## Available Actions:
- Navigation: GotoLocation <object_id>
- Object Interaction: PickupObject <object_id>, PutObject <object_id, receptacle_id>
- Container Actions: OpenObject <object_id>, CloseObject <object_id>
- Appliance Actions: ToggleObjectOn <object_id>, ToggleObjectOff <object_id>
- Object Modification: SliceObject <object_id>
- Perception: GetObjectInFrame <x, y>
- Task Completion: stop

## Instructions:
Generate the NEXT SINGLE ACTION to progress toward completing the task. Consider:
1. What you need to do to complete the task
2. What actions you've already taken
3. If you do not have object IDs, use GetObjectInFrame with image coordinates
4. Use GotoLocation <object_id> to move toward target objects before interacting

GetObjectInFrame expects normalized image coordinates in [0, 1].

Respond with ONLY the action in this format:
{"action": "ActionName", "object_id": "ObjectId|x|y|z"} (if object needed)
{"action": "GetObjectInFrame", "x": 0.64, "y": 0.40}
{"action": "ActionName"} (if no object needed)
{"action": "PutObject", "object_id": "AlarmClock|+01.65|+00.80|-01.28", "receptacle_id": "CounterTop|+02.10|+00.90|+01.50"} (if placing an object)


Next action is:
"""
        if image_available:
            prompt += "\n(An image of the current scene is attached.)\n"
        return prompt

    def update_action_history(self, action, success, error=None, result_object_id=None):
        self.completed_actions.append({
            'action': action.get('action'),
            'object_id': action.get('object_id'),
            'x': action.get('x'),
            'y': action.get('y'),
            'result_object_id': result_object_id,
            'success': success,
            'error': error,
            'timestamp': len(self.completed_actions),
        })

    def query_vlm(self, system_prompt, user_prompt, image=None):
        try:
            messages = [{"role": "system", "content": system_prompt}]
            if image is None:
                messages.append({"role": "user", "content": user_prompt})
            else:
                encoded = self._encode_image(image)
                user_content = [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                ]
                messages.append({"role": "user", "content": user_content})

            data = {
                "model": getattr(self.args, 'model', 'openai/gpt-5'),
                "messages": messages,
                "max_tokens": getattr(self.args, 'max_tokens', 10000),
                "temperature": getattr(self.args, 'temperature', 0.6),
                "top_p": getattr(self.args, 'top_p', 1.0),
                "frequency_penalty": getattr(self.args, 'frequency_penalty', 0.0),
                "presence_penalty": getattr(self.args, 'presence_penalty', 0.0),
            }

            response = self._post_request(data)
            content = response['choices'][0]['message']['content']

            self.log("VLM RESPONSE:")
            self.log(content)
            self.log("-" * 50)
            return content
        except Exception as e:
            error_msg = f"[ERROR] Unexpected error calling VLM: {e}"
            self.log(f"{error_msg}")
            print(error_msg)
        return None

    def _encode_image(self, image):
        if isinstance(image, Image.Image):
            img = image
        else:
            img = Image.fromarray(image)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return encoded

    def _get_api_key(self):
        from models.model.llm import API_KEY
        return API_KEY

    def _post_request(self, data):
        import requests
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._get_api_key()}", "Content-Type": "application/json"},
            json=data,
        )
        response.raise_for_status()
        return response.json()
