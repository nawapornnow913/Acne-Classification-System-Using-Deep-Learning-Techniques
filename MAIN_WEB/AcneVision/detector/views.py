import os
import base64
import numpy as np
import cv2
import io
import csv 
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from ultralytics import YOLO
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'best.pt')
model = YOLO(MODEL_PATH)

def load_acne_db_from_csv():
    csv_path = os.path.join(settings.BASE_DIR, 'data_1', 'acne_care.csv')
    db = {}
    yolo_classes = ['whitehead', 'blackhead', 'papules', 'pustule', 'nodule']
    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                full_name = row['type'].strip()
                char = row['acne_char'].strip()
                how_to = row['how_to'].strip()
                for cls in yolo_classes:
                    if cls.lower() in full_name.lower():
                        short_name = full_name.split(' (')[0].split(' หรือ')[-1].strip()
                        db[cls] = {
                            'short_name': short_name,
                            'full_name': full_name,
                            'char': char,
                            'how_to': how_to
                        }
                        break
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการโหลด CSV: {e}")
    return db

ACNE_DB = load_acne_db_from_csv()

try:
    font = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 22)
except:
    font = ImageFont.load_default()

def home(request):
    return render(request, 'index.html')

def detect_acne(request):
    if request.method == 'POST':
        image_b64 = request.POST.get('image')
        if image_b64:
            format, imgstr = image_b64.split(';base64,') 
            img_data = base64.b64decode(imgstr)
            img_pil_original = Image.open(io.BytesIO(img_data)).convert('RGB')
            img_np = np.array(img_pil_original)
            img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            results = model.predict(source=img_cv2, conf=0.1, save=False, show=False)
            r = results[0]

            summary = []
            if len(r.boxes) == 0:
                summary.append("""
                <div style='text-align: center; color: #1f2937; font-size: 16px; font-weight: 800; padding: 24px; 
                            background: linear-gradient(135deg, rgba(255, 255, 255, 0.4) 0%, rgba(255, 255, 255, 0.1) 100%); 
                            backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); 
                            border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.6); 
                            font-family: "Prompt", sans-serif; text-transform: uppercase; letter-spacing: 1px; 
                            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);'>
                    ไม่พบสิวบนใบหน้า (CLEAR SKIN)
                </div>
                """)
                annotated_frame = img_cv2
            else:
                detected_classes = r.boxes.cls.tolist()
                class_counts = Counter(detected_classes)
                
                for cls_id, count in class_counts.items():
                    eng_name = r.names[int(cls_id)].lower()
                    acne_info = ACNE_DB.get(eng_name, {
                        'full_name': eng_name, 
                        'char': 'ไม่มีข้อมูลในระบบ', 
                        'how_to': 'ไม่มีข้อมูลในระบบ',
                        'short_name': eng_name
                    })
                    
                    html_snippet = f"""
                    <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.45) 0%, rgba(255, 255, 255, 0.15) 100%); 
                                backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); 
                                padding: 24px; border-radius: 20px; margin-bottom: 16px; 
                                border: 1px solid rgba(255, 255, 255, 0.6); box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05); 
                                font-family: 'Prompt', sans-serif; transition: all 0.3s; color: #1f2937;">
                        
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; border-bottom: 1px dashed rgba(31, 41, 55, 0.2); padding-bottom: 12px;">
                            <div style="font-size: 16px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; color: #111827;">
                                {acne_info['full_name']}
                            </div>
                            <div style="background: rgba(255, 255, 255, 0.7); color: #1f2937; font-size: 12px; font-weight: 800; 
                                        padding: 6px 14px; border-radius: 20px; letter-spacing: 1px; border: 1px solid rgba(255, 255, 255, 0.8); box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                                {count} จุด
                            </div>
                        </div>
                        
                        <div style="font-size: 14px; color: #1f2937; margin-bottom: 12px; line-height: 1.6;">
                            <span style="font-weight: 800; color: #000000;">ลักษณะ:</span> {acne_info['char']}
                        </div>
                        <div style="font-size: 14px; color: #1f2937; line-height: 1.6;">
                            <span style="font-weight: 800; color: #000000;">คำแนะนำ:</span> {acne_info['how_to']}
                        </div>
                    </div>
                    """
                    summary.append(html_snippet)
                
                annotated_frame = r.plot(labels=False, conf=False)
                img_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    eng_name = r.names[cls_id].lower()
                    acne_name = ACNE_DB.get(eng_name, {}).get('short_name', eng_name)
                    x1, y1, x2, y2 = box.xyxy[0].tolist() 
                    text_x, text_y = x1, max(0, y1 - 25)
                    
                    draw.text((text_x+1, text_y+1), acne_name, font=font, fill=(50, 50, 50)) 
                    draw.text((text_x, text_y), acne_name, font=font, fill=(255, 255, 255)) 
                
                annotated_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            _, buffer = cv2.imencode('.jpg', annotated_frame)
            result_img_str = base64.b64encode(buffer).decode('utf-8')
            final_image_base64 = f"data:image/jpeg;base64,{result_img_str}"

            return JsonResponse({'success': True, 'image': final_image_base64, 'summary': summary})

    return JsonResponse({'success': False, 'error': 'Invalid request'})