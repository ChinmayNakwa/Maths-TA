# backend/utils/process_and_embed.py

import time
import random
import os
import subprocess
import json
import cv2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pathlib import Path
import math
import re
from youtube_transcript_api import YouTubeTranscriptApi

# Import our configuration and vector store connection
from ..config import settings
from ..database.astra_db_connection import get_vector_store
from .cloudinary_uploader import upload_image_to_cloudinary

# --- Directory Setup ---
VIDEO_DOWNLOAD_DIR = Path("temp_videos")
IMAGE_OUTPUT_DIR = Path(settings.IMAGE_OUTPUT_DIR)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- HELPER FUNCTIONS ---
def sanitize_filename(name: str) -> str:
    """Removes illegal characters from a string to make it a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def get_transcript_for_video(video_id: str):
    """Fetches and formats the transcript for a given YouTube video ID."""
    try:
        # Correctly call the method from the imported library
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ""
        for item in transcript_list:
            start_time = int(item['start']); h,m = divmod(start_time, 3600); m,s = divmod(m, 60)
            timestamp = f"[{h:02d}:{m:02d}:{s:02d}]"
            full_transcript += f"{timestamp} {item['text']}\n"
        return full_transcript
    except Exception as e:
        print(f"  - Could not fetch transcript for {video_id}: {e}")
        return None

# There should only be one definition of this function
def extract_and_upload_keyframes(video_path: Path, video_id: str, interval_seconds=60) -> dict:
    """Extracts keyframes, uploads them, and returns a map of {timestamp: url}."""
    print(f"  - Extracting and uploading keyframes for {video_id}...")
    keyframe_url_map = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  - Error opening video file.")
        return keyframe_url_map
    
    # Calculate video duration for logging
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        print("  - Error: Video frame rate is zero.")
        cap.release()
        return keyframe_url_map
    video_length_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_rate)
    print(f"    - Video duration: ~{video_length_sec // 60} minutes.")
        
    frame_interval = int(frame_rate * interval_seconds)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            timestamp_sec = int(frame_count / frame_rate)
            public_id = f"video_{video_id}_frame_{timestamp_sec}"
            temp_image_path = IMAGE_OUTPUT_DIR / f"{public_id}.png"
            
            print(f"    - Processing frame for timestamp {timestamp_sec}s...")
            cv2.imwrite(str(temp_image_path), frame)
            
            image_url = upload_image_to_cloudinary(temp_image_path, public_id=public_id)
            if image_url:
                print(f"      - Success! URL: {image_url}")
                keyframe_url_map[timestamp_sec] = image_url

        frame_count += 1
        
    cap.release()
    print("  - Keyframe processing complete.")
    return keyframe_url_map # Make sure to return the map

def check_if_video_exists_in_db(video_id: str, vector_store) -> bool:
    """Checks if chunks for a given video_id already exist in AstraDB."""
    if not vector_store or not hasattr(vector_store, 'collection'): return False
    try:
        results = vector_store.collection.find(filter={"metadata.video_id": video_id}, projection={"_id": 1}, limit=1)
        return bool(results and results.get("documents"))
    except Exception as e:
        print(f"  - Could not check DB. Will process. Error: {e}")
        return False

# --- MAIN PROCESSING FUNCTION ---
def process_downloaded_videos():
    """Iterates through downloaded videos, processes them, and adds to the vector store."""
    print("\n--- Starting Downloaded Video Processing ---")
    
    if not VIDEO_DOWNLOAD_DIR.exists() or not any(VIDEO_DOWNLOAD_DIR.iterdir()):
        print("No videos found in the 'temp_videos' directory. Skipping.")
        return

    print("Initializing vector store...")
    vector_store = get_vector_store()

    print("Fetching playlist info to map filenames to video IDs...")
    command = ["yt-dlp", "--flat-playlist", "--print", "%(id)s;%(title)s", settings.YOUTUBE_PLAYLIST_URL]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        video_infos_raw = [line.split(';', 1) for line in result.stdout.strip().split('\n') if line]
        video_info_map = {sanitize_filename(title): {'id': vid, 'url': f"https://www.youtube.com/watch?v={vid}", 'original_title': title} for vid, title in video_infos_raw}
    except Exception as e:
        print(f"FATAL: Could not get playlist info to process videos. Error: {e}")
        return
    
    videos_to_process = sorted(list(VIDEO_DOWNLOAD_DIR.glob("*.mp4")))
    total_videos = len(videos_to_process)

    # Iterate through the downloaded files
    for i, video_path in enumerate(videos_to_process):
        sanitized_title = video_path.stem
        video_details = video_info_map.get(sanitized_title)
        
        if not video_details:
            print(f"  - Could not find metadata for video file: {video_path.name}. Skipping.")
            continue
            
        video_id = video_details['id']
        video_url = video_details['url']
        original_title = video_details['original_title']
        
        print(f"\n--- Processing local file {i+1}/{total_videos}: {video_path.name} ---")

        if check_if_video_exists_in_db(video_id, vector_store):
            print(f"Data for video '{sanitized_title}' already in DB. Skipping.")
            continue

        try:
            full_transcript = get_transcript_for_video(video_id)
            if not full_transcript: 
                print("  - No transcript found. Skipping.")
                continue
            
            keyframe_urls = extract_and_upload_keyframes(video_path, video_id)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
            chunks = text_splitter.split_text(full_transcript)
            docs = []
            for chunk in chunks:
                match = re.search(r'\[(\d{2}):(\d{2}):(\d{2})\]', chunk)
                chunk_start_time = 0
                if match:
                    h, m, s = map(int, match.groups())
                    chunk_start_time = h * 3600 + m * 60 + s
                closest_keyframe_time = math.floor(chunk_start_time / 60) * 60
                image_url = keyframe_urls.get(closest_keyframe_time, "")
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": "video", "title": original_title, "video_id": video_id,
                        "video_url": video_url, "image_path": image_url,
                        "start_time_sec": chunk_start_time
                    }
                ))
            
            if docs:
                vector_store.add_documents(docs)
                print(f"  - Added {len(docs)} video chunks to vector store.")
            
        except Exception as e:
            print(f"  - !!! FAILED to process {sanitized_title}: {e}")
        
        finally:
            # Add delay after each video processing attempt (success or fail)
            if i < total_videos - 1:
                delay = random.uniform(5, 15)
                print(f"\n---Waiting for {delay:.2f} seconds before next video... ---")
                time.sleep(delay)

    print("\n--- All Video Processing Complete ---")

if __name__ == '__main__':
    process_downloaded_videos()