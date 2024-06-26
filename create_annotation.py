import json
import os

# val.txt 파일 경로 설정
val_file = 'datasets/kaist-rgbt/val.txt'
annotation_output = 'datasets/kaist-rgbt/annotations/KAIST_annotation.json'  # 특정 경로에 저장

# 이미지 크기 설정 (모든 이미지가 동일한 크기라고 가정)
image_height = 512
image_width = 640

# 이미지 정보 수집
with open(val_file, 'r') as f:
    val_lines = f.readlines()

images = []
annotations = []
annotation_id = 0

for idx, line in enumerate(val_lines):
    image_path = line.strip()
    im_info = {
        "id": idx,
        "im_name": image_path,
        "height": image_height,
        "width": image_width
    }
    images.append(im_info)

    # 어노테이션 파일 경로 추출 (예: 이미지 경로에서 어노테이션 경로로 변환)
    annotation_file = image_path.replace('images', 'annotations').replace('.jpg', '.txt')
    print(annotation_file)  # 경로 확인용 출력

    # 어노테이션 파일 읽기 (여기서는 예제로 각 라인이 'class x y w h' 형식이라고 가정)
    if os.path.exists(annotation_file):
        with open(annotation_file, 'r') as af:
            for line in af:
                parts = line.strip().split()
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])

                ann_info = {
                    "id": annotation_id,
                    "image_id": idx,
                    "category_id": class_id,
                    "bbox": [x, y, w, h],
                    "height": h,
                    "occlusion": 0,
                    "ignore": 0
                }
                annotations.append(ann_info)
                annotation_id += 1

# JSON 파일 생성
kaist_annotation = {
    "info": {
        "dataset": "KAIST Multispectral Pedestrian Benchmark",
        "url": "https://soonminhwang.github.io/rgbt-ped-detection/",
        "related_project_url": "http://multispectral.kaist.ac.kr",
        "publish": "CVPR 2015"
    },
    "info_improved": {
        "sanitized_annotation": {
            "publish": "BMVC 2018",
            "url": "https://li-chengyang.github.io/home/MSDS-RCNN/",
            "target": "files in train-all-02.txt (set00-set05)"
        },
        "improved_annotation": {
            "url": "https://github.com/denny1108/multispectral-pedestrian-py-faster-rcnn",
            "publish": "BMVC 2016",
            "target": "files in test-all-20.txt (set06-set11)"
        }
    },
    "images": images,
    "annotations": annotations,
    "categories": [
        {"id": 0, "name": "person"},
        {"id": 1, "name": "cyclist"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "person?"}
    ]
}

# 파일 저장
with open(annotation_output, 'w') as f:
    json.dump(kaist_annotation, f, indent=4)
