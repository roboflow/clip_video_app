# Functions to handle CLIP
import pickle
import numpy as np
from typing import Dict, List
from collections import defaultdict
import os
import requests
import cv2
import base64
from PIL import Image
from io import BytesIO

# Load frame embeddings
def load_frame_embeddings(save_frame_embeddings_path: str):
    if os.path.exists(save_frame_embeddings_path):
        with open(save_frame_embeddings_path, 'rb') as f:
            frame_embeddings = pickle.load(f)
    else:
        frame_embeddings = {}

    return frame_embeddings

# Save frame embeddings
def save_frame_embeddings(save_frame_embeddings_path: str, frame_embeddings: Dict[str, List[float]]):
    # Save the updated frame embeddings
    with open(save_frame_embeddings_path, 'wb') as f:
        pickle.dump(frame_embeddings, f)

    return True


# Get CLIP text embeddings
def get_clip_text_embeddings(objects: List[str], saved_embeddings_path: str, api_key: str):
    """
    Get CLIP embeddings for a list of object names.
    
    Args:
        objects (List[str]): List of object names.
        saved_embeddings_path (str): Path to save or load existing embeddings.
        
    Returns:
        Dict: Dictionary containing CLIP embeddings.
    """
    
    # Load existing embeddings if available
    embeddings = {}
    if os.path.exists(saved_embeddings_path):
        with open(saved_embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
    
    # Generate new embeddings for objects that don't have one
    new_objects = [obj for obj in objects if obj not in embeddings]

    # Loop through text and embed
    for txt_object in new_objects:
        print(txt_object)
        # CLIP Endppoint
        request_endpoint = "http://localhost:9001" + "/clip/embed_text?api_key=" + api_key
        # Payload
        payload = {
            "body": request_endpoint.split('=')[1],
            "text": txt_object
            }
        # Request
        data = requests.post(
            request_endpoint, json=payload
            ).json()
        # Grab embeddings
        embedding = data["embeddings"]
        
        # Add to dict
        embeddings[txt_object] = embedding

    # Filter any extra left over embeddings out
    keys_to_remove = [key for key in embeddings.keys() if key not in objects]
    for key in keys_to_remove:
        del embeddings[key]

    # Save the updated embeddings
    with open(saved_embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

# Image Embedding
def get_clip_image_embeddings(frame, api_key: str):
    # Get CLIP embedding
    # Convert the array to a PIL Image
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_data = Image.fromarray(frame)

    buffer = BytesIO()
    image_data.save(buffer, format="JPEG")
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # CLIP Endppoint
    request_endpoint = "http://localhost:9001" +  "/clip/embed_image?api_key=" + api_key
    # Payload
    payload = {
        "body": request_endpoint.split('=')[1],
        "image": {"type": "base64", "value": image_data},
        }

    # Request
    data = requests.post(
        request_endpoint, json=payload
        ).json()
    # Grab embeddings
    embedding = data["embeddings"]
    return embedding

# Frame / object similarity
def get_most_similar_objects(frame_embedding: np.ndarray, object_embeddings: Dict[str, np.ndarray], 
                             historical_scores: defaultdict, history_length: int = 1, top_n: int = 3) -> List[str]:
    """
    Find the N most similar objects based on CLIP embeddings with smoothing.

    Args:
        frame_embedding (np.ndarray): CLIP embedding of the frame.
        object_embeddings (Dict[str, np.ndarray]): Dictionary of object embeddings.
        historical_scores (defaultdict): Historical similarity scores for smoothing.
        history_length (int): Number of past frames to consider for smoothing.
        top_n (int): Number of top similar objects to return.

    Returns:
        List[str]: List of N most similar objects.
    """
    smoothed_similarities = {}
    frame_embedding = np.squeeze(frame_embedding)
    
    for obj, obj_embedding in object_embeddings.items():
        obj_embedding = np.squeeze(obj_embedding)
        dot_product = np.dot(frame_embedding, obj_embedding)
        norm_frame = np.linalg.norm(frame_embedding)
        norm_obj = np.linalg.norm(obj_embedding)
        similarity = dot_product / (norm_frame * norm_obj)
        
        # Update history and calculate smoothed score
        if len(historical_scores[obj]) >= history_length:
            historical_scores[obj].popleft()
        historical_scores[obj].append(similarity)
        
        smoothed_similarity = sum(historical_scores[obj]) / len(historical_scores[obj])
        smoothed_similarities[obj] = smoothed_similarity

    sorted_similarities = sorted(smoothed_similarities.items(), key=lambda x: x[1], reverse=True)
    most_similar_objects = [(obj, score) for obj, score in sorted_similarities[:top_n]]

    return most_similar_objects, smoothed_similarities