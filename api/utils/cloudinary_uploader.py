import cloudinary
import cloudinary.uploader
from backend.config import settings
from pathlib import Path

cloudinary.config(
    cloud_name = settings.CLOUDINARY_CLOUD_NAME,
    api_key = settings.CLOUDINARY_API_KEY,
    api_secret = settings.CLOUDINARY_API_SECRET,
    secure = True
)

def upload_image_to_cloudinary(image_path: Path, public_id: str) -> str:
    """
    Uploads a local image file to Cloudinary and returns its secure URL.
    Deletes the local file after a successful upload.
    
    Args:
        image_path: The local path to the image file.
        public_id: A unique identifier for the image in Cloudinary.
                   This helps prevent duplicates.
    
    Returns:
        The secure URL of the uploaded image, or an empty string on failure.
    """
    if not image_path.exists():
        return ""
    
    try:
        upload_result = cloudinary.uploader.upload(
            str(image_path),
            public_id = public_id,
            overwrite = False,
            folder = "maths_ta",
            timeout=30
        )

        image_path.unlink()

        return upload_result.get('secure_url', '')
    
    except Exception as e:
        print(f" - Error uploading {public_id} to Cloudinary: {e}")
        return ""
    
# if __name__ == "__main__":
#     path = Path(r"C:\Users\chinm\Desktop\GitHub\Maths-TA\output_images\book_ProbabilityForComputerScientists_page_1.png")
#     result = upload_image_to_cloudinary(path, "test_page_1")
#     print(f"Upload result: {result}")

