import cv2
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from threading import Thread
import yaml
from pathlib import Path
from typing import Any, Dict
import clip_functions
from collections import deque

# Flask App
app = Flask(__name__, template_folder='../templates', static_folder='../static')
socketio = SocketIO(app)

# Open webpage when running script
open_browser = True

# Flags
play_video = False
# Paths where embeddings are saved.
save_frame_embeddings_path = ''
save_object_embeddings_path = ''

# Variables
frame_number = 0
successful_frame_number = 0 # js code emit back the latest frame number that was successfully processed
frames_processed = []
video_path = ""  # Replace with your video path
api_key = ''
string_list = []
object_embeddings = {}
frame_embeddings = {}
historical_scores = None
thread = None

# CV cap
cap = None

# Read config file
def read_config(file_path: Path) -> Dict[str, Any]:
    """
    Reads a YAML configuration file and returns it as a dictionary.
    
    Args:
        file_path (Path): The path to the YAML configuration file.
        
    Returns:
        Dict[str, Any]: A dictionary containing the configuration data.
    """
    global video_path, api_key, string_list
    
    absolute_path = Path(file_path).resolve()
    with open(absolute_path, 'r') as f:
        config_data = yaml.safe_load(f)
    # Set API and Video paths
    api_key = config_data['roboflow_api_key']
    video_path = config_data['video_path']
    string_list = config_data['CLIP']


# Play video thread
def play_video_function():
    global play_video, cap, frame_number, frames_processed, historical_scores

    # Open video
    if cap is None:
        cap = cv2.VideoCapture(video_path)

    # For smoothing out similarity scores
    if historical_scores is None:
        historical_scores = {obj: deque(maxlen=1000) for obj in object_embeddings.keys()}
    print("Starting video processing... {}".format(play_video))
    while play_video:

        ret, frame = cap.read()
        frame_number += 1
        if not ret:
            print("Video ended.")
            break
        print('Server processing frame {}'.format(frame_number))

        # Convert frame to base64 and emit to client
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Grab frame key
        frame_key = f"frame_{frame_number}"
        # Check if frame embedding already exists
        if frame_key not in frame_embeddings:
            # Embed frame in CLIP and add to dict
            frame_embedding = clip_functions.get_clip_image_embeddings(frame, api_key)
            frame_embeddings[frame_key] = frame_embedding

            # Save the updated frame embeddings
            clip_functions.save_frame_embeddings(save_frame_embeddings_path, frame_embeddings)
        else:
            frame_embedding = frame_embeddings[frame_key]

        # Calculate similarity between frame and ob jects
        most_similar_objects, all_object_scores = clip_functions.get_most_similar_objects(frame_embedding, object_embeddings, historical_scores)

        # emit similarity to client
        number_of_lines = len(all_object_scores)
        line_values = [value for value in all_object_scores.values()]
        line_names = [title for title in all_object_scores.keys()]
        line_data = [{"title": title, "value": value} for title, value in zip(line_names, line_values)]

        # Wait until we get confirmation that the frame previous was processed
        print(f"Waiting for frame {successful_frame_number} to be processed... with {frame_number}")
        while successful_frame_number != frame_number-1:
            # print('waiting')
            socketio.sleep(.05)

        # Check if frame has already been processed
        if frame_number in frames_processed:
            # Emit
            socketio.emit('frame', {
                'frame': frame_base64,
                'frame_number': frame_number,
                'set_line': True
            })
        else:
            # Emit
            socketio.emit('frame', {
                'frame': frame_base64,
                'frame_number': frame_number,
                'lines': line_data
            })
            # Add to frames processed
            frames_processed.append(frame_number)

        # Sleep so stuff works
        socketio.sleep(.05)

def reset_state():
    global play_video, cap, frame_number, frames_processed, successful_frame_number, thread
    # Call stop
    stop_function()
    play_video = False
    frame_number = 0
    # Add any other state variables that need to be reset
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        emit('frame', {
            'frame': frame_base64,
            'frame_number': frame_number,
            'set_frame': True
        })
    
    # Reset variables
    frames_processed = []
    historical_scores = None
    thread = None  # Reset thread to allow a new one to be created later
    successful_frame_number = 0


@socketio.on('reset')
def handle_reset():
    print("Received 'reset' event from client.")
    reset_state()

@app.route('/')
def index():
    """Render the main index.html page."""
    return render_template('index.html')

@socketio.on('client_error')
def handle_client_error(json):
    print("Received 'client_error' event from client.")
    error_message = json.get('error', 'Unknown error')
    
    # Log the error
    print(f"Client-side error: {error_message}")

@socketio.on('successful_frame_number')
def handle_success_frame(msg):
    global successful_frame_number
    successful_frame_number = int(msg['frame_number'])
    print('Received successful frame number: {}'.format(successful_frame_number))

@socketio.on('start')
def handle_start(msg):
    global thread, play_video
    print("Received 'start' event from client.")
    play_video = True

    if thread is None or not thread.is_alive():
        thread = Thread(target=play_video_function)
        thread.start()

def stop_function():
    global play_video, thread
    print("Received 'stop' event from client.")
    play_video = False
    thread = None  # Reset thread to allow a new one to be created later

@socketio.on('stop')
def handle_stop(msg):
    stop_function()

@socketio.on('set_frame')
def handle_set_frame(msg):
    global cap, frame_number
    frame_number = int(msg['frame_number'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        emit('frame', {
            'frame': frame_base64,
            'frame_number': frame_number,
            'set_frame': True
        })

if __name__ == '__main__':
    # Read config file
    config_data = read_config('config.yaml')
    # Calculate Text Embeddings
    save_object_embeddings_path = "embeddings/{}_text.pkl".format(video_path.split('/')[-1].split('.')[0])
    object_embeddings = clip_functions.get_clip_text_embeddings(string_list, save_object_embeddings_path, api_key)

    # Load existing frame embeddings (if available)
    save_frame_embeddings_path = "embeddings/{}_frames.pkl".format(video_path.split('/')[-1].split('.')[0])
    frame_embeddings = clip_functions.load_frame_embeddings(save_frame_embeddings_path)

    # Run
    socketio.run(app, debug=True)
