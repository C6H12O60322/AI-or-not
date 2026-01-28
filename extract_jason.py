import json
import os
import re

# Detect the message type based on the UI components shown to the user.
# See https://discord.com/developers/docs/interactions/message-components#what-is-a-component
COMPONENTS_FOR_INITIAL_AND_VARIATION = set(
    ['U1', 'U2', 'U3', 'U4', '⟳', 'V1', 'V2', 'V3', 'V4'])
COMPONENTS_FOR_UPSCALE = set(
    ['Make Variations', 'Upscale to Max', 'Light Upscale Redo'])

def get_message_type(message):
    """Figures out the message type based on the UI components displayed."""
    for components in message["components"]:
        for component in components["components"]:
            if component["label"] in COMPONENTS_FOR_INITIAL_AND_VARIATION:
                # For (very few) messages that are supposedly initial or variation requests, the content indicates
                # that they are actually upscale requests. We will just put these aside.
                if "Upscaled" in message["content"]:
                    return "INCONCLUSIVE"
                return "INITIAL_OR_VARIATION"
            elif component["label"] in COMPONENTS_FOR_UPSCALE:
                return "UPSCALE"
    return "TEXT_MESSAGE"

def get_prompt(message):
    """Extracts the prompt from the message content, which is located between double stars."""
    content = message["content"]
    # Replace newlines with spaces; makes the regex below work.
    content = content.replace("\n", " ")
    # Find the text enclosed by two consecutive stars.
    BETWEEN_STARS = "\\*\\*(.*?)\\*\\*"
    match = re.search(BETWEEN_STARS, content)
    if match:
        return match.group()[2:-2]  # Exclude the stars.
    

def remove_urls(prompt):
    """Prompts can include both text and images; this method removes the prompt image URLs."""
    URL = "<https[^<]*>?\s"
    matches = re.findall(URL, prompt)
    for match in matches:
        prompt = prompt.replace(match, "")
    return prompt

filepaths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        filepaths.append(os.path.join(dirname, filename))
print(f"Found {len(filepaths)} files.")


prompts = []
for filepath in filepaths:
    with open(filepath, "r") as f:
        content = json.load(f)
        for single_message_list in content["messages"]:
            assert len(single_message_list) == 1
            message = single_message_list[0]
            message_type = get_message_type(message)

            if message_type not in ["INITIAL_OR_VARIATION", "UPSCALE"]:
                continue  # Ignore direct text messages.

            prompt = get_prompt(message)
            if not prompt:
                continue  # Discard malformed messages.
                
            # The goal of this dataset is to learn *text prompts*, so remove any image prompts.
            text_prompt = remove_urls(prompt)
            prompts.append(text_prompt)
            
print(f"Extracted {len(prompts)} text prompts.")