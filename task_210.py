import os
import shutil
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def detect_changes(before_path, after_path):
    before = cv2.imread(before_path)
    after = cv2.imread(after_path)

    before = cv2.resize(before, (after.shape[1], after.shape[0]))

    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    gray_before = cv2.GaussianBlur(gray_before, (5, 5), 0)
    gray_after = cv2.GaussianBlur(gray_after, (5, 5), 0)

    diff_abs = cv2.absdiff(gray_before, gray_after)
    _, mask_abs = cv2.threshold(diff_abs, 20, 255, cv2.THRESH_BINARY)

    score, diff_ssim = ssim(gray_before, gray_after, full=True)
    diff_ssim = (1 - diff_ssim) * 255
    diff_ssim = diff_ssim.astype(np.uint8)
    diff_ssim = cv2.GaussianBlur(diff_ssim, (5, 5), 0)
    _, mask_ssim = cv2.threshold(diff_ssim, 20, 255, cv2.THRESH_BINARY)

    combined_mask = cv2.bitwise_or(mask_abs, mask_ssim)

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # Final blur to reduce tiny noise areas
    combined_mask = cv2.GaussianBlur(combined_mask, (3, 3), 0)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = after.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        contour_area = cv2.contourArea(cnt)
        aspect_ratio = w / float(h + 1e-5)
        solidity = contour_area / float(area + 1e-5)

        # Extract the region from the mask
        roi_mask = combined_mask[y:y+h, x:x+w]
        change_pixels = cv2.countNonZero(roi_mask)
        change_ratio = change_pixels / float(area + 1e-5)

        # Final smart condition
        if (
            area > 120 and
            0.3 < aspect_ratio < 3.5 and
            solidity > 0.45 and
            change_ratio > 0.15  # key line: at least 15% pixels must have changed
        ):
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return output


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = os.listdir(input_folder)

    before_images = [f for f in files if '~2' not in f and f.lower().endswith(('.jpg', '.png'))]

    for before in before_images:
        base_name = before.rsplit('.', 1)[0]
        after_name = f"{base_name}~2.jpg"
        output_name = f"{base_name}~3.jpg"

        before_path = os.path.join(input_folder, before)
        after_path = os.path.join(input_folder, after_name)
        output_before_copy = os.path.join(output_folder, before)
        output_result_path = os.path.join(output_folder, output_name)

        if os.path.exists(after_path):
            print(f"Processing: {before} & {after_name}")

            # ✅ Copy the before image into output folder
            if not os.path.exists(output_before_copy):
                shutil.copy2(before_path, output_before_copy)

            result = detect_changes(before_path, after_path)
            cv2.imwrite(output_result_path, result)
        else:
            print(f"After image not found for: {before}")


if __name__ == "__main__":
    input_folder = "input-images"
    output_folder = "output_folder"
    process_folder(input_folder, output_folder)

