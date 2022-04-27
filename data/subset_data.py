import json
from re import S
from tqdm import tqdm
import shutil
from pathlib import Path

# Download full coco2017 (images+annotations) train+val first

def get_subset(mode):
  src_img_parent = '{}2017/'.format(mode)
  src_anno_parent = 'annotations/'

  final_path = 'coco2017{}_subset/'.format(mode)
  dst_img_parent = final_path + 'images/'
  dst_anno_parent = final_path + 'annotations/'

  shutil.rmtree(final_path, ignore_errors=True)
  Path(dst_img_parent).mkdir(parents=True, exist_ok=True)
  Path(dst_anno_parent).mkdir(parents=True, exist_ok=True)

  # 'person': 1, 'car': 3, 'cat': 17, 'bottle': 44, 'chair': 62
  id_cls = [3, 17]

 
  with open("{}/instances_{}2017.json".format(src_anno_parent, mode), "r") as f:
    data = json.load(f)

    annotations = []
    images = []
    img_id = []

    dict_images = {img['id']:img for img in data['images']}

    for anno in tqdm(data['annotations']):
      if anno['category_id'] in id_cls:
        annotations.append(anno)
        
        if anno['image_id'] not in img_id:
          img_id.append(anno['image_id'])
          images.append(dict_images[anno['image_id']])
          shutil.copyfile(src_img_parent + dict_images[anno['image_id']]['file_name'], dst_img_parent + dict_images[anno['image_id']]['file_name'])
    data['images'] = images
    data['annotations'] = annotations

    with open(dst_anno_parent + 'instances_{}2017.json'.format(mode), 'w') as f:
      json.dump(data, f)

if __name__ == "__main__":
  get_subset('train')
  get_subset('val')