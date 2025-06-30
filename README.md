🧠 Objective
Detect and highlight differences between before and after images of the same scene by drawing bounding boxes around changed regions.

📁 Folder Structure
arduino
Copy
Edit
.
├── task_2_code.py        ← Python script to run the detection
├── task_2_output/        ← Folder containing input and output images
│   ├── 1.jpg             ← Input "before" image
│   ├── 1~3.jpg           ← Output: "after" image with bounding boxes
│   ├── 2.jpg
│   ├── 2~3.jpg
│   └── ...
⚠️ X~2.jpg (original after image) is used internally but not stored in the final output.


🚀 How to Run
### 1. Install dependencies (once):

```bash
pip install -r requirements.txt
```

### 2. Place images in the task_2_output/ folder:
Before image: X.jpg
After image: X~2.jpg

### 3. Run the script:

```bash
python task_2_code.py
```

The script will generate:
X~3.jpg: Annotated image showing the changes
It will also ensure X.jpg is retained in the folder

🧪 How It Works
Converts both images to grayscale and applies Gaussian blur.

Computes differences using:
cv2.absdiff (raw pixel difference)
SSIM (structural difference)

Combines both masks and applies morphological operations to reduce noise.

Detects contours and draws bounding boxes using smart filtering:
Area threshold
Aspect ratio
Solidity
Pixel change ratio inside each box

📦 Dependencies
See requirements.txt:
opencv-python
numpy
scikit-image

