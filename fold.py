"""
Args : 
    - number of folds (in this problem, we decide 5-fold as k-fold validation)
    - index of the fold
    
Returns : 
    - datasets/kaist_rgbt/train_{idx}_{fold}.txt
    - datasets/kaist_rgbt/val_{idx}_{fold}.txt 
    - datasets/KAIST_annotation.json
"""

import os
import json
import math 
import argparse

class_map = {
    "person": 0,
    "cyclist": 1,
    "people": 2,
    "person?": 3
}

class KFoldValidation:
    def __init__(self, args):
        self.dataset = args.dataset 
        self.root_path = os.path.join(os.environ.get("PWD"), self.dataset)

        self.train_txt = os.path.join(self.root_path, "train-all-04.txt")
        with open(self.train_txt, 'r') as f:
            self.file_list = f.readlines()

        # Options for K-cross validation
        self.k_cross_val = args.k
        self.k_index = args.idx 
        self.imgs_per_fold = math.floor(len(self.file_list) / self.k_cross_val)
        print(self.imgs_per_fold)

        # Files
        self.train_list = []
        self.train_txt_file = os.path.join(self.root_path, f"train_{self.k_index}_{self.k_cross_val}.txt")
        self.val_list = []
        self.val_txt_file = os.path.join(self.root_path, f"val_{self.k_index}_{self.k_cross_val}.txt")
        
        self.annotation_file = os.path.join(self.root_path, "annotations", f"annot_{self.k_index}_{self.k_cross_val}.json")
        self.annotation_file_example = os.path.join(self.root_path, "annotations", "example_annot.json")
        self.annotations = None
        self.annot_images = []
        self.annot_annotations = []
        self.annot_categories = {}
        
        for line, file_name in enumerate(self.file_list):
            if line >= self.imgs_per_fold * self.k_index and line <= self.imgs_per_fold * (self.k_index + 1):
                self.val_list.append(file_name)
            else:
                self.train_list.append(file_name)        

    def generate_train_file(self):
        with open(self.train_txt_file, "w+") as f:
            for line in self.train_list:
                f.write(line)

    def generate_val_file(self):
        with open(self.val_txt_file, "w+") as f:
            for line in self.val_list:
                f.write(line)

    def generate_annotations(self):
        with open(self.annotation_file_example, "r") as example:
            baseline = json.load(example)
        self.annot_categories = baseline["categories"]

        # Generate image entries
        for idx, line in enumerate(self.val_list):
            tmp_image = {
                "id": idx,
                "im_name": line.strip(),
                "height": 512,
                "width": 640
            }
            self.annot_images.append(tmp_image)

        # Generate annotation entries
        import xml.etree.ElementTree as ET
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        self.xml_path = os.path.join(self.root_path, "train", "labels-xml")
        self.annotation_id = 0 
        for idx, line in enumerate(self.val_list):
            file_name = os.path.basename(line.strip())
            xml_file = file_name.replace(".jpg", ".xml")
            
            with open(os.path.join(self.xml_path, xml_file)) as f:
                tree = ET.parse(f)
                root = tree.getroot()

            for data in root:
                bbox = []
                if data.tag == "object":
                    for element in data:
                        if element.tag == "name":
                            name = element.text
                        if element.tag == "bndbox":
                            for xywh in element:
                                bbox.append(int(xywh.text))
                        if element.tag == "occlusion":
                            occlusion = int(element.text)
                            
                    tmp_annotation = {
                        "id": self.annotation_id,
                        "image_id": idx,
                        "category_id": class_map[name],
                        "bbox": bbox,
                        "height": bbox[-1],
                        "occlusion": occlusion,
                        "ignore": 0
                    }
                    self.annotation_id += 1
                    self.annot_annotations.append(tmp_annotation)
    
    # Merge dictionaries into one JSON file
    def merge_to_json(self):
        self.annotations = {
            "images": self.annot_images,
            "annotations": self.annot_annotations,
            "categories": self.annot_categories
        }

        with open("KAIST_annotation.json", "w") as f:
            json.dump(self.annotations, f)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--idx", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="kaist-rgbt")
    opt = parser.parse_args()
    
    k_fold_val = KFoldValidation(opt)
    
    k_fold_val.generate_train_file()
    k_fold_val.generate_val_file()
    k_fold_val.generate_annotations()
    k_fold_val.merge_to_json()
    
    print("Done!")
