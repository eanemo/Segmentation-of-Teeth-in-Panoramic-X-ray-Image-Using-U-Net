import os
import numpy as np
import argparse
from pycocotools.coco import COCO
import cv2


TEETH_COLORS = [(0,0,0), (255,0,0), (0,255,0), (255,255,0), (0,0,255), (255,0,255), (0,255,255)]

######### MAIN #########
def main():
    parser = argparse.ArgumentParser(description='Parse requirements and model paths.')
    parser.add_argument('--annotationpath', help='coco annotation path in json format', required=True)
    parser.add_argument('--savepath', help='save directory location in project dir', required=True)
    args = vars(parser.parse_args())

    annotation_path = args['annotationpath']
    save_folder = args['savepath']

    # Store directory
    if not os.path.isdir(save_folder):
        os.mkdir(str(save_folder))
        print('Create output directory: {}'.format(save_folder))


    cocoann = COCO(annotation_path)
    cat_ids = cocoann.getCatIds()
    print("Categories IDs:", cat_ids)
    imgIds = cocoann.getImgIds()
    print("ImageIDs:", imgIds)
    anns_ids = cocoann.getAnnIds(imgIds=imgIds, catIds=cat_ids, iscrowd=None)
    print("Annotation IDs:", anns_ids)
    anns = cocoann.loadAnns(anns_ids)
    print("Annotations:", anns)

    for img_id in imgIds:       # for each image
        img = cocoann.loadImgs(img_id)[0]
        print("Image id in for:", img_id, img['file_name'])
        anns_img = np.zeros((img['height'], img['width']))       # RGB blank image

        for ann in anns:        # for each annotation
            if ann['image_id'] == img_id:       # check annotations for the image
                ann_mask = cocoann.annToMask(ann) * ann['category_id']
                anns_img = np.maximum(anns_img, ann_mask)
                print(ann)

        print("Mask image unique:", np.unique(ann_mask))
        mask_name = os.path.splitext(img['file_name'])[0]
        mask_path = os.path.join(save_folder, mask_name+".png")
        print(mask_path)
        cv2.imwrite(mask_path, anns_img)


def annotation_to_colormask(mask, color):
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3))

    for w in range(width):
        for h in range(height):
            if mask[h,w] == 1:
                color_mask[h,w,:] = color

    return color_mask





##############  MAIN  ##############
if __name__ == '__main__':
    main()
