import numpy as np
import random
from PIL import Image

def resize_image_for_mosaic(img, output_size, max_size=800):
    """(Pillow) 이미지의 짧은 쪽을 output_size에 맞추고 비율을 유지하며 리사이즈"""
    w, h = img.size
    scale = output_size / min(h, w)
    
    if max(h, w) * scale > max_size:
        scale = max_size / max(h, w)
    
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    return resized_img, scale

def apply_mosaic_augmentation(first_dict, dataset_list, output_size, max_size):
    """
    (Pillow) 4개 이미지를 사전 리사이즈 후, 고정 그리드에 합치기만 수행하는 함수
    """
    
    # 1. 4개의 이미지 선택 및 Pillow로 로드
    indices = [first_dict["_mosaic_idx"]] + random.sample(range(len(dataset_list)), 3)
    mosaic_dicts = [dataset_list[i] for i in indices]
    # Image.open을 사용하여 Pillow 이미지 객체로 로드
    images = [Image.open(d["file_name"]).convert("RGB") for d in mosaic_dicts]

    # 2. 사전 리사이즈
    resized_data = [resize_image_for_mosaic(img, output_size, max_size) for img in images]
    resized_images = [data[0] for data in resized_data]
    scales = [data[1] for data in resized_data]

    # 3. 고정 그리드 배치를 위한 캔버스 생성 (Pillow 사용)
    max_h = max(img.height for img in resized_images)
    max_w = max(img.width for img in resized_images)
    # 채움 색상으로 회색(114, 114, 114) 사용
    mosaic_img = Image.new('RGB', (max_w * 2, max_h * 2), (114, 114, 114))

    # 4. 고정 그리드에 이미지 배치 및 어노테이션/관계 결합
    final_annotations = []
    final_relations = []
    object_offset = 0

    for i, (img, d, scale) in enumerate(zip(resized_images, mosaic_dicts, scales)):
        # 배치 위치 계산 (고정 그리드)
        x_offset, y_offset = (max_w, 0) if i % 2 else (0, 0)
        if i > 1: y_offset = max_h
            
        # Pillow의 paste 메서드로 이미지 붙여넣기
        mosaic_img.paste(img, (x_offset, y_offset))

        if "annotations" in d:
            annos = d["annotations"]
            
            for anno in annos:
                new_anno = anno.copy()
                bbox = np.array(new_anno["bbox"], dtype=float) * scale
                bbox[[0, 2]] += x_offset
                bbox[[1, 3]] += y_offset
                new_anno["bbox"] = bbox.tolist()
                final_annotations.append(new_anno)
        
        if "relations" in d and len(d["relations"]) > 0:
            relations = np.array(d["relations"])
            relations[:, :2] += object_offset
            final_relations.extend(relations.tolist())
        
        object_offset += len(annos)

    # 5. 최종 결과를 NumPy 배열로 변환하여 반환
    #    DetrDatasetMapper의 후속 로직은 NumPy 배열을 기대하므로,
    #    Pillow(RGB) 이미지를 NumPy(BGR) 배열로 변환합니다.
    final_img_np = np.array(mosaic_img, dtype=np.uint8)
    final_img_bgr = final_img_np[:, :, ::-1] # RGB -> BGR

    return {
        "image_data": final_img_bgr,
        "height": final_img_bgr.shape[0],
        "width": final_img_bgr.shape[1],
        "annotations": final_annotations,
        "relations": final_relations,
    }