# KratosGPT![godofwar4blogroll-1523487184188](https://github.com/klally36/Kratos-GPT/assets/174218325/7ebd6a02-8e61-4bf8-83a4-582458c5b04a)

## Overview
This is KratosGPT! This project leverages artificial intelligence (AI) to play God of War: Ragnarok. By utilizing a LLM agent, an image captioning model, and YOLOv8 for object detection, this AI system controls the main character, Kratos, guiding his actions within the game. Users can try out this functionality for free using a Cohere trial API Key. This README provides comprehensive instructions on implementing this AI system in any game, promising an adventure filled with excitement and triumph.




## Prerequisites
Before proceeding, ensure you have the following prerequisites:
#### 1. Cohere Trial API Key: Sign up for a Cohere account to obtain your trial API key, allowing access to Cohere's Language models.
#### 2. Game and Tools: Install God of War: Ragnarok on your system and ensure all requirements from requirements.txt are installed, including image captioning models and YOLOv8.
#### 3. GPT-4 Text and Image API (Optional): Optionally, if you have access, the GPT-4 Text and Image API can enhance AI capabilities by using image input directly with GPT-4.

## Project Structure
The project is structured as follows:


#### 1. Creating a Custom YOLOv8 Model
#### 2. Preprocessing Training Videos
#### 3. Creating the Dataset
#### 4. Training the Object Detection Model
#### 5. Modifying the LLM Agent and Play Functions
Each step is outlined in detail below.


##  Step 1: Creating a Custom YOLOv8 Model<a name='1'></a>
To begin, a custom YOLOv8 model is forged to identify characters, enemies, and NPCs within the game. This involves recording videos of each character individually and training the model to detect them.


To create the custom YOLOv8 model, we require individual recordings of each character. These recordings should ideally encompass diverse environments, various armor/skin combinations, and different character poses, if feasible. Utilizing a game's photo mode, if available, can facilitate capturing the character from multiple perspectives.

To streamline the process, instead of manually annotating bounding boxes for each character frame by frame, we employ object detection techniques. By utilizing an object detection model, characters within the videos are automatically identified, and the resulting bounding box coordinates serve as training data. Given that most in-game characters possess humanoid features, they are typically detected as humans, allowing us to utilize these coordinates effectively. This approach significantly minimizes manual labor.

## Step 2: Preprocessing Training Videos<a name='2'></a>
After capturing videos for each character, we proceed to preprocess them by eliminating frames where no detections occur. This procedure serves to refine the data, ensuring that only frames where the model accurately detects the character's position are retained. Utilizing the "preprocess_training_video" function found within the revered "preprocess.py" script, we cleanse our videos of imperfections. Additionally, this process aids in determining the appropriate threshold value for character detection, a crucial step for dataset creation.
```sh
from functions.preprocess import preprocess_training_video

# Specify the path to your video file
video_path = "training_videos/atreus_training_video.mp4"

# Call the function to perform object detection on the video
preprocess_training_video(video_path)
```

## Step 3: Creating the Dataset<a name='3'></a>
Having purified our videos, we must now create a robust dataset. Armed with the paths to our videos and the determined threshold values, we invoke the formidable "create_dataset" function housed within the "dataset.py" script. This yields a dataset primed to serve as training data for our object detection model. Based on the threshold that yielded optimal character identification during preprocessing, we compile a list of video paths paired with their respective threshold values. The resulting dataset encapsulates the essential information required for training our object detection model.
```sh
from functions.dataset import create_dataset

# Set the paths to video_paths and set the values in threshold_list
video_paths = ['preprocessed_videos/Draugr.mp4', 'preprocessed_videos/Atreus.mp4', 'preprocessed_videos/Kratos.mp4']
threshold_list = [0.6, 0.7, 0.9]

# Create the dataset
create_dataset(video_paths, threshold_list)
```

## Step 4: Training the Object Detection Model<a name='4'></a>
With the dataset ready, we proceed to train our object detection model using the YOLOv8 architecture. This model is trained on the dataset created earlier to recognize the main character, enemy characters, and other NPCs based on the provided training data. The "train_model" function, found in the "train.py" script, facilitates this process. Setting the path to the dataset YAML file, determining the number of training epochs, and specifying the image size are crucial steps that influence the model's performance. Invocation of the "train_model" function with these parameters initiates the training process. By doing this, the model evolves to accurately perceive the characteristics of our characters and adversaries.
```sh
from functions.train import train_model

# Set the path to the dataset YAML file
dataset_yaml = "dataset.yaml"

# Set the number of training epochs and image size
num_epochs = 100

# Set imgsz 
img_size = 640

# Train the model using the dataset
train_model(dataset_yaml, num_epochs, img_size)
```

## Step 5: Modifying the LLM Agent and Play Functions<a name='5'></a>
Now, we proceed to integrate the trained object detection model into the game control mechanism. This involves modifying the "agent.py" and "play.py" scripts to accommodate the new model and enable effective control of the main character by the AI. In "agent.py," the Cohere trial API key is added to the "llm_agent" function. In "play.py," adjustments are made based on the final object detection model's name and the corresponding keys responsible for character movement, special moves, and attacks. The "llm_agent" function is utilized to control the character's actions, including attacks and special moves. Depending on the detected positions of the main character and enemy characters, the LLM Agent determines appropriate actions for the character to execute.

Note that two object detection models are used in the "play_game" function—one for the main character and another for enemies—to accommodate limited training data and diverse character types.
**Example of moving Player**
```sh
# Move Kratos towards the nearest person
if distance_x < 0:
    keys.directKey("a")
    sleep(1)
    keys.directKey("a", keys.key_release)
elif distance_x > 0:
    keys.directKey("d")
    sleep(1)
    keys.directKey("d", keys.key_release)

if distance_y < 0:
    keys.directKey("w")
    sleep(1)
    keys.directKey("w", keys.key_release)
elif distance_y > 0:
    keys.directKey("s")
    sleep(1)
    keys.directKey("s", keys.key_release)
else:
# Move the camera if no person is detected
keys.directMouse(-20, 0)
sleep(0.04)
```

**Example of doing attacks or special moves**
```sh            
# Select an action
action = llm_agent(screen) 
if action == "light attack":
    # Left mouse click (attack)
    keys.directMouse(buttons=keys.mouse_lb_press)
    sleep(0.5)
    keys.directMouse(buttons=keys.mouse_lb_release)
elif action == "heavy attack":
    # Right mouse click (attack)
    keys.directMouse(buttons=keys.mouse_rb_press)
    sleep(0.5)
    keys.directMouse(buttons=keys.mouse_rb_release)
elif action == "dodge back":
    # Move in the opposite direction to the NPC
    direction = random.choice(["w", "a", "s", "d"])
    keys.directKey(direction)
    sleep(0.04)
    keys.directKey(direction, keys.key_release)
    # Press the space bar twice
    keys.directKey('0x39')
    sleep(0.04)
    keys.directKey("0x39", keys.key_release)
    sleep(0.04)
    keys.directKey('0x39')
    sleep(0.04)
    keys.directKey("0x39", keys.key_release)
```

## ️ Playing the Game with AI
### ️ The Dance of Victory ⚔️
With our preparations complete, we are ready to initiate gameplay and utilize our enhanced capabilities. The "play_game" function, found within the "play.py" script, facilitates guiding our character through the virtual world.

The YOLOv8 model detects the positions of our character and enemies, while the "Key.py" script provides guidance based on sentdex's CyberPython Tutorial. The LLM agent determines the character's actions and maneuvers, showcasing intelligence in decision-making.

The following code sets our character in motion, enabling navigation through the virtual realm with precision and efficacy:
```sh
from functions.play import play_game
import time

# sleep for some time to give me time to open the game
time.sleep(100)

# Call the function to perform object detection on the screen
play_game()
```

## Conclusion
Congratulations on completing the journey through AI-Plays-God-of-War! This endeavor has involved traversing the depths of artificial intelligence, blending magic and technology to achieve triumph. Together, we have utilized the power of YOLOv8, trained a robust object detection model, and activated an LLM agent to guide our actions.




