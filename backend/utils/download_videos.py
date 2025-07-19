# backend/utils/download_videos.py

import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re

# Import config to get the playlist URL
from backend.config import settings

# --- Directory Setup ---
VIDEO_DOWNLOAD_DIR = Path("temp_videos")
VIDEO_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Removes illegal characters from a string to make it a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def download_video_task(video_info: dict) -> dict:
    """
    Downloads a single video using yt-dlp with retries. Does NOT use cookies.
    This function is designed to be run in a thread pool.
    """
    title = video_info['title']
    sanitized_title = sanitize_filename(title)
    video_path = VIDEO_DOWNLOAD_DIR / f"{sanitized_title}.mp4"

    # Check 1: If the final file already exists, we're done.
    if video_path.exists():
        print(f"  [SKIP] Video already exists: {video_path.name}")
        video_info['status'] = 'exists'
        return video_info

    command = [
        "yt-dlp",
        "-q", "--no-warnings",  # Quiet mode
        "-f", "bv[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best", # Prefer mp4
        "--recode-video", "mp4", # Fallback to recode if needed
        "-o", str(video_path),
        video_info['url']
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        print(f"  [ATTEMPT {attempt + 1}/{max_retries}] Downloading: {video_path.name}")
        try:
            # Run the command. Timeout after 10 minutes.
            subprocess.run(command, check=True, capture_output=True, timeout=600)
            print(f"  [SUCCESS] Finished download: {video_path.name}")
            video_info['status'] = 'downloaded'
            return video_info
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            error_message = e.stderr.decode('utf-8', 'ignore') if hasattr(e, 'stderr') and e.stderr else str(e)
            # Get the last line of the error, which is usually the most specific
            last_error_line = error_message.strip().splitlines()[-1]
            print(f"  [FAIL {attempt + 1}] {video_path.name} | Error: {last_error_line}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Wait 5, 10 seconds
                print(f"  - Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  [GIVING UP] All download attempts failed for {video_path.name}")
                video_info['status'] = 'failed'
                return video_info
    
    video_info['status'] = 'failed' # Should not be reached, but for safety
    return video_info


def main():
    """Main function to download all videos from the playlist."""
    playlist_url = settings.YOUTUBE_PLAYLIST_URL
    if not playlist_url or "YOUR_YOUTUBE_PLAYLIST_URL" in playlist_url:
        print("Error: Youtube playlist URL not set in config.py.")
        return

    print(f"--- Starting Parallel Video Download ---")
    print(f"Playlist URL: {playlist_url}")

    # 1. Get playlist video information
    print("\nFetching playlist information...")
    command = ["yt-dlp", "--flat-playlist", "--print", "%(id)s;%(title)s", playlist_url]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        video_infos_raw = [line.split(';', 1) for line in result.stdout.strip().split('\n') if line]
        video_infos = [{'id': vid, 'title': title, 'url': f"https://www.youtube.com/watch?v={vid}"} for vid, title in video_infos_raw]
        print(f"Found {len(video_infos)} videos in the playlist.")
    except subprocess.CalledProcessError as e:
        print(f"FATAL: Error fetching playlist info with yt-dlp: {e.stderr}")
        return

    # 2. Run downloads in parallel
    successful_downloads = 0
    failed_downloads = 0
    # Use max_workers to control how many downloads run at once. 5 is a safe number.
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_video = {executor.submit(download_video_task, info): info for info in video_infos}
        for future in as_completed(future_to_video):
            result_info = future.result()
            if result_info['status'] in ['downloaded', 'exists']:
                successful_downloads += 1
            else:
                failed_downloads += 1
    
    print("\n--- Download Phase Complete ---")
    print(f"Successfully downloaded/found: {successful_downloads} videos.")
    print(f"Failed to download: {failed_downloads} videos.")
    print(f"All available videos are now in the '{VIDEO_DOWNLOAD_DIR}' directory.")


if __name__ == '__main__':
    main()