import cv2
import numpy as np
import os
from datetime import datetime
import sys
from tkinter import Tk, filedialog, messagebox

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[0] - br[0]) ** 2))
    heightB = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[0] - bl[0]) ** 2))
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    denoised = cv2.fastNlMeansDenoising(sharpened)
    return denoised

def scan_document(input_path):
    try:
        image = cv2.imread(input_path)
        if image is None:
            messagebox.showerror("错误", "无法读取图片文件！")
            return

        orig = image.copy()
        ratio = image.shape[0] / 500.0
        resized = cv2.resize(image, (int(image.shape[1]/ratio), 500))

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            warped = orig
        else:
            screenCnt = screenCnt.reshape(4, 2) * ratio
            warped = four_point_transform(orig, screenCnt)

        enhanced = enhance_image(warped)

        # 输出文件夹
        output_dir = "scans"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        out_folder = os.path.join(output_dir, f"{base_name}_{timestamp}")
        os.makedirs(out_folder, exist_ok=True)

        cv2.imwrite(os.path.join(out_folder, "01_original_warped.jpg"), warped)
        cv2.imwrite(os.path.join(out_folder, "02_enhanced_grayscale.jpg"), enhanced)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(out_folder, "03_binary.jpg"), binary)
        color_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(out_folder, "RECOMMENDED_enhanced.jpg"), color_enhanced)

        messagebox.showinfo("成功", f"处理完成！\n\n输出文件夹：{out_folder}\n\n推荐使用：RECOMMENDED_enhanced.jpg")
        
    except Exception as e:
        messagebox.showerror("错误", f"处理失败：{str(e)}")

if __name__ == "__main__":
    # 图形界面选择文件
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="选择要扫描的文档图片",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if file_path:
        scan_document(file_path)
    else:
        messagebox.showwarning("取消", "未选择图片")