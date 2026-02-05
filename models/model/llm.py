import os
import json
import requests
from datetime import datetime

API_KEY = os.getenv("API_KEY")

class LLMAgent:
    """
    LLM Agent for ALFRED task execution using OpenRouter API
    """
    
    def __init__(self, args):
        self.args = args
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.log_method = None  # Will be set by caller
        
    def set_log_method(self, log_method):
        """Set logging method from caller"""
        self.log_method = log_method
    
    def log(self, message):
        """Log message if log method is available"""
        if self.log_method:
            self.log_method(message)
        else:
            print(message)
    
    def get_subgoals_from_scene(self, task_desc, scene_info):
        """
        Generate subgoals from task description and scene metadata
        """
        from models.prompts import SUBGOAL_PROMPT, SAFETY_SUBGOAL_PROMPT

        user_prompt = f"""
Task: {task_desc}

Scene Information:
- Agent Position: {scene_info['agent']['position']}
- Agent Rotation: {scene_info['agent']['rotation']}

Available Objects:
{self.format_objects_for_prompt(scene_info['objects'], filter_visible=False)}

Safety Constraints:
{self.load_safety_constraint(scene_info['objects'])}

Generate subgoals for the given task using only the provided objects. Format your output in JSON as specified above. DO NOT include any explanations or additional text. DO NOT wrap the JSON in markdown.
"""
        # self.log("=" * 50)
        # self.log("SUBGOAL GENERATION PROMPT:")
        # self.log(user_prompt)
        # self.log("=" * 50)
        self.log("Generating subgoals from scene...")
        subgoals = self.query_llm(SAFETY_SUBGOAL_PROMPT, user_prompt)
        return subgoals

    def generate_plan(self, subgoals, scene_info, goto=False):
        """
        Generate action plan from subgoals and scene information
        """
        from models.prompts import ACTION_SEQ_PROMPT, ACTION_SEQ_PROMPT_GOTO
        
        # Create prompt for LLM
        user_prompt = self.create_prompt(subgoals, scene_info, goto=goto)
        
        # Log the prompt
        # self.log("=" * 50)
        # self.log("PLAN GENERATION PROMPT:")
        # self.log(user_prompt)
        # self.log("=" * 50)

        system_prompt = ACTION_SEQ_PROMPT
        if goto:
            system_prompt = ACTION_SEQ_PROMPT_GOTO
        plan_text = self.query_llm(system_prompt, user_prompt)
        plan = self.parse_llm_response(plan_text)
        
        self.log(f"Generated plan with {len(plan)} actions")
        for i, action in enumerate(plan):
            self.log(f"  {i+1}. {action}")
        
        return plan

    def generate_safety_constraints(self, scene_info, scene_name=None, notes=None):
        """
        Derive safety constraints for the provided scene snapshot using the LLM.
        """
        from models.prompts import SAFETY_CONSTRAINT_PROMPT

        if scene_info is None:
            raise ValueError("scene_info is required to generate safety constraints")

        canonical_rules = self.load_safety_constraint(scene_info.get('objects', []))
        snapshot = json.dumps(
            scene_info,
            indent=2,
            sort_keys=True,
            default=self._json_default,
        )

        prompt = []
        if scene_name:
            prompt.append(f"Scene Name: {scene_name}")
        prompt.append("Scene Metadata (sanitized metadata):")
        prompt.append(snapshot)
        prompt.append("Canonical Safety Rules relevant to observed object types:")
        prompt.append(canonical_rules)
        if notes:
            prompt.append("User Notes:")
            prompt.append(str(notes))

        user_prompt = "\n".join(prompt) + "\n"

        response = self.query_llm(SAFETY_CONSTRAINT_PROMPT, user_prompt)
        if not response:
            return None

        try:
            parsed = json.loads(response)
            return parsed
        except json.JSONDecodeError:
            self.log("Failed to parse safety constraint response as JSON; returning raw content.")
            return response
    
    def create_prompt(self, subgoals_json, scene_info, goto=False):
        """
        Create prompt for LLM with subgoals and scene information
        """
        # Parse subgoals with improved handling for markdown format
        subgoals = self.extract_subgoals(subgoals_json)

        prompt = f"""
## Current Agent Status:
- Position: {scene_info['agent']['position']}
- Rotation: {scene_info['agent']['rotation']}
- Holding: {scene_info['inventoryObjects'] if scene_info['inventoryObjects'] else 'Nothing'}

## Subgoals to Achieve (in order):
"""
        
        # Add numbered subgoals for clarity
        if subgoals:
            for i, subgoal in enumerate(subgoals, 1):
                prompt += f"{i}. {subgoal}\n"
        else:
            prompt += "No valid subgoals found - will generate basic plan\n"
        
        prompt += "\n## Relevant Objects in Scene:\n"
        prompt += self.format_objects_for_prompt(scene_info['objects'], filter_visible=False)
            
        if goto:
            prompt += """
## Available Actions:
- Navigation: GotoLocation <object_id>
- Object Interaction: PickupObject <object_id>, PutObject <object_id, object_id>
- Container Actions: OpenObject <object_id>, CloseObject <object_id>
- Appliance Actions: ToggleObjectOn <object_id>, ToggleObjectOff <object_id>
- Object Modification: SliceObject <object_id>
- Task Completion: stop

## Planning Rules:
1. **Achieve subgoals in the given order** - each subgoal builds on the previous ones
2. **Navigation first** - Navigate to objects before interacting with them
3. **Object interactions** - Pick up objects with PickupObject before moving them
4. **State changes** - Use appropriate actions to change object states
5. **Exact object IDs** - Use the exact object IDs from the subgoals
6. **Find objects first** - Navigate around to find objects if needed
7. **Complete all subgoals** - Don't stop until all subgoals are achieved
8. **PutObject requires two object_ids** - the object being placed and the receptacle
9. **Pick up an object in a closed receptacle** - If the receptacle is openable and closed, open it first before picking up the object inside. 

## Output Format:
Generate a JSON array of actions that will achieve the subgoals in order:
[
    {"action": "GotoLocation", "object_id": "AlarmClock|+01.65|+00.80|-01.28"},
    {"action": "PickupObject", "object_id": "AlarmClock|+01.65|+00.80|-01.28"},
    {"action": "GotoLocation", "object_id": "DeskLamp|-01.31|+01.23|-00.99"},
    {"action": "ToggleObjectOn", "object_id": "DeskLamp|-01.31|+01.23|-00.99"},
    {"action": "GotoLocation", "object_id": "CounterTop|+02.10|+00.90|+01.50"},
    {"action": "PutObject", "object_id": "AlarmClock|+01.65|+00.80|-01.28", "receptacle_id": "CounterTop|+02.10|+00.90|+01.50"},
    {"action": "stop"}
]

Generate the action sequence to achieve all subgoals:
"""
            return prompt

        prompt += """
## Available Actions:
- Navigation: MoveAhead, MoveBack, MoveLeft, MoveRight, RotateLeft, RotateRight, LookUp, LookDown
- Object Interaction: PickupObject <object_id>, PutObject <object_id, object_id>
- Container Actions: OpenObject <object_id>, CloseObject <object_id>
- Appliance Actions: ToggleObjectOn <object_id>, ToggleObjectOff <object_id>
- Object Modification: SliceObject <object_id>

## Planning Rules:
1. **Achieve subgoals in the given order** - each subgoal builds on the previous ones
2. **Navigation first** - Navigate to objects before interacting with them
3. **Object interactions** - Pick up objects with PickupObject before moving them
4. **State changes** - Use appropriate actions to change object states
5. **Exact object IDs** - Use the exact object IDs from the subgoals
6. **Find objects first** - Navigate around to find objects if needed
7. **Complete all subgoals** - Don't stop until all subgoals are achieved
8. **PutObject requires two object_ids** - the object being placed and the receptacle


## Output Format:
Generate a JSON array of actions that will achieve the subgoals in order:

[
    {"action": "MoveAhead"},
    {"action": "RotateRight"},
    {"action": "PickupObject", "object_id": "AlarmClock|+01.65|+00.80|-01.28"},
    {"action": "MoveLeft"},
    {"action": "ToggleObjectOn", "object_id": "DeskLamp|-01.31|+01.23|-00.99"},
    {"action": "MoveAhead"},
    {"action": "MoveAhead"},
    {"action": "PutObject", "object_id": "AlarmClock|+01.65|+00.80|-01.28", "receptacle_id": "CounterTop|+02.10|+00.90|+01.50"},
    {"action": "stop"}
]

Generate the action sequence to achieve all subgoals:
"""
        return prompt

    def extract_subgoals(self, subgoals_json):
        """Extract subgoals handling markdown format"""
        if not subgoals_json:
            self.log("No subgoals provided by LLM; continuing without structured subgoals")
            raise ValueError("No subgoals provided by LLM")

        subgoals_json = str(subgoals_json).strip()
        if "```json" in subgoals_json:
            # Find the start and end of the JSON block
            start_marker = "```json"
            end_marker = "```"

            start_idx = subgoals_json.find(start_marker)
            if start_idx != -1:
                # Move past the start marker
                start_idx += len(start_marker)
                
                # Find the end marker after the start
                end_idx = subgoals_json.find(end_marker, start_idx)
                if end_idx != -1:
                    # Extract the JSON content
                    json_content = subgoals_json[start_idx:end_idx].strip()
                else:
                    # If no closing ```, take everything after ```json
                    json_content = subgoals_json[start_idx:].strip()
            else:
                # Fallback: try to extract after ```json
                json_content = subgoals_json.split("```json", 1)[-1].strip()
                if json_content.endswith("```"):
                    json_content = json_content[:-3].strip()
        else:
            # No markdown code blocks, use the whole response
            json_content = subgoals_json

        data = json.loads(json_content)
        return data.get('subgoals', [])

    def query_llm(self, system_prompt, user_prompt):
        """
        Query LLM via OpenRouter API
        """
        try:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }
            
            data = {
                "model": getattr(self.args, 'model', 'deepseek/deepseek-chat-v3.1'),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": getattr(self.args, 'max_tokens', 10000),
                "temperature": getattr(self.args, 'temperature', 0.6),
                "top_p": getattr(self.args, 'top_p', 1.0),
                "frequency_penalty": getattr(self.args, 'frequency_penalty', 0.0),
                "presence_penalty": getattr(self.args, 'presence_penalty', 0.0)
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            # Raise exception for HTTP errors
            response.raise_for_status()
            
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            
            # Log the response
            self.log("LLM RESPONSE:")
            self.log(content)
            self.log("-" * 50)
            return content

        except Exception as e:
            error_msg = f"[ERROR] Unexpected error calling LLM: {e}"
            self.log(f"{error_msg}")
            print(error_msg)

        return None

    def parse_llm_response(self, response_text):
        """
        Parse LLM response into action list with robust error handling
        """
        if not response_text:
            return [{'action': 'stop'}]
            
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                plan = json.loads(json_str)
                
                # Validate that actions are properly formatted
                validated_plan = []
                for action in plan:
                    if isinstance(action, dict) and 'action' in action:
                        validated_plan.append(action)
                
                return validated_plan if validated_plan else [{'action': 'stop'}]
                
        except Exception as e:
            self.log(f"Error parsing JSON from LLM response: {e}")

        return [{'action': 'stop'}]

    def format_objects_for_prompt(self, objects, filter_visible=False):
        """
        Format objects list into a string for prompt inclusion
        
        Args:
            objects: List of object dictionaries from scene_info['objects']
            filter_visible: If True, only include visible objects
        
        Returns:
            Formatted string ready for prompt inclusion
        """
        if not objects:
            return "No objects available.\n"
        
        prompt_section = ""
        
        for obj in objects:
            # Filter by visibility if requested
            if filter_visible and not obj.get('visible'):
                continue
                
            # Start with object type and ID
            line = f"- {obj['objectType']} ({obj['objectId']}): "
            
            # Collect object properties
            properties = []
            
            if obj['pickupable'] and not obj['isPickedUp']:
                properties.append("pickupable")
            elif obj['isPickedUp']:
                properties.append("being held")
            if obj['receptacle']:
                properties.append("receptacle")
            if obj['openable']:
                properties.append(f"openable ({'open' if obj['isOpen'] else 'closed'})")
            if obj['toggleable']:
                if "Candle" in obj['objectId']:
                    properties.append(f"toggleable ({'it is on'})")
                else:
                    properties.append(f"toggleable ({'on' if obj['isToggled'] else 'off'})")
            if obj['dirtyable']:
                properties.append(f"{'dirty' if obj['isDirty'] else 'clean'}")
            if obj['cookable']:
                properties.append(f"{'cooked' if obj['isCooked'] else 'uncooked'}")
            if obj.get('isSliced'):
                properties.append("sliced")
            if obj['temperature'] != 'RoomTemp':
                properties.append(f"temperature: {obj['temperature']}")
            if obj['parentReceptacles']:
                properties.append(f"in receptacle: {', '.join(obj['parentReceptacles'])}")
            if obj['receptacleObjectIds']:
                properties.append(f"contains {len(obj['receptacleObjectIds'])} items, including:")
                for rec_id in obj['receptacleObjectIds']:
                    properties.append(f" {rec_id};")

            line += ", ".join(properties) if properties else "no special properties"
            line += f" at {obj['position']}\n"
            
            # Add visibility indicator if not filtering by visible
            if not filter_visible:
                visibility = "visible" if obj.get('visible') else "not visible"
                line += f" [{visibility}]"
                
            prompt_section += line + "\n"
        
        return prompt_section

    def load_safety_constraint(self, objects):
        # Load safety constraints from JSON file
        with open('safety_rules_object.json', 'r') as f:
            safety_data = json.load(f)

        # Generate safety constraints based on object properties
        constraints = set()
        
        seen = set()
        for obj in objects:
            line = "- "
            if obj["objectType"] in safety_data and obj["objectType"] not in seen:
                for rule in safety_data[obj["objectType"]]:
                    constraints.add(rule)
                seen.add(obj["objectType"])
        line += "\n ".join(constraints) if constraints else "no safety constraints"
        return line

    @staticmethod
    def _json_default(value):
        if isinstance(value, datetime):
            return value.isoformat()
        return repr(value)
