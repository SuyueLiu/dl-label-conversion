import os
import pandas as pd
import shutil
import argparse
import cv2
from tqdm import tqdm


class Epic2Darknet():
    def __init__(self, org_ann_path, org_names_path, org_dataset_dir, out_dir, mode, iou_thres):
        self.org_ann_pth = org_ann_path
        self.org_names_path = org_names_path
        self.org_dataset_dir = org_dataset_dir
        self.out_dir = out_dir
        self.mode = mode
        self.iou_thres = iou_thres

    @staticmethod
    def write_to_file(file_path, boxes_info):
        """ Write the content to file"""

        for i in range(len(boxes_info)):
            temp = ''
            for j in boxes_info[i]:
                temp += str(j) + ' '

            with open(file_path, 'a') as f_obj:
                f_obj.write(temp.strip(' ') + '\n')

    def add_extra_files(self, root_dir):
        def reformat_path(path):
            if path.split('/')[0] == '..':
                path = path.replace('..', '.')
            return path

        dataset_name = root_dir.split('/')[-1]

        # write _train.txt
        train_txt_path = os.path.join(root_dir, f"{dataset_name}_train.txt")
        train_imgs_dir = os.path.join(root_dir, 'images/train')
        train_imgs = os.listdir(train_imgs_dir)
        train_imgs_dir = reformat_path(train_imgs_dir)
        with open(train_txt_path, 'a') as f1:
            for train_img in train_imgs:
                f1.write(os.path.join('./images/train', train_img) + '\n')

        # write _val.txt
        val_txt_path = os.path.join(root_dir, f"{dataset_name}_val.txt")
        val_imgs_dir = os.path.join(root_dir, 'images/val')
        val_imgs = os.listdir(val_imgs_dir)
        val_imgs_dir = reformat_path(val_imgs_dir)
        with open(val_txt_path, 'a') as f2:
            for val_img in val_imgs:
                f2.write(os.path.join('./images/val', val_img) + '\n')

        # write .names
        out_names_path = os.path.join(root_dir, f"{dataset_name}.names")
        names = pd.read_csv(self.org_names_path)['class_key'].tolist()
        with open(out_names_path, 'a') as f:
            if os.path.exists(out_names_path):
                pass
            else:
                for name in names:
                    f.write(name + '\n')

        # write .data
        datafile_path = os.path.join(root_dir, f"{dataset_name}.data")
        with open(datafile_path, 'a') as f4:
            f4.write('classes=' + str(len(names)) + '\n')
            f4.write('train=' + reformat_path(train_txt_path) + '\n')
            f4.write('valid=' + reformat_path(val_txt_path) + '\n')
            f4.write('names=' + reformat_path(out_names_path))

    @staticmethod
    def move_images(new_dir, org_dir, image_mask, data_folder_name):
        for img_name in tqdm(image_mask, desc=f"moving images ({data_folder_name})"):
            img_path = os.path.join(org_dir, img_name)
            new_img_path = os.path.join(new_dir, data_folder_name + '-' + img_name)
            if os.path.exists(new_img_path):
                continue
            else:
                shutil.copy(img_path, new_img_path)

    @staticmethod
    def bbox_iou(box1, box2):
        """
        Return the IoU of two bounding boxes that have the coordinate <x_c, y_c, w, h>

        """
        box1 = [float(x) for x in box1]
        box2 = [float(x) for x in box2]

        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = max(b1_x1, b2_x1)
        inter_rect_y1 = max(b1_y1, b2_y1)
        inter_rect_x2 = min(b1_x2, b2_x2)
        inter_rect_y2 = min(b1_y2, b2_y2)

        inter_w = inter_rect_x2 - inter_rect_x1
        inter_h = inter_rect_y2 - inter_rect_y1
        inter_area = 0 if inter_h < 0 or inter_w < 0 else inter_h * inter_w

        # Union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        IoU = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return IoU

    @staticmethod
    def coor_transform(bbox, img_size):
        """The bounding boxes in epic-55 have the format (top-left y, top-left x, h, w)
           The bounding boxes in Darknet have the format (center x, center y, w, h)
        """
        y1, x1, h, w = bbox
        img_h, img_w, _ = img_size

        # normed coordinates
        x_c = (x1 + w / 2) / img_w
        y_c = (y1 + h / 2) / img_h
        w /= img_w
        h /= img_h

        return [('%.6f' % i) for i in [x_c, y_c, w, h]]

    @staticmethod
    def make_dirs(new_data_dir, mode):

        if mode == 'train':
            img_dir = os.path.join(new_data_dir, 'images', mode)
            os.makedirs(img_dir, exist_ok=True)

            label_dir = os.path.join(new_data_dir, 'labels', mode)
            os.makedirs(label_dir, exist_ok=True)

        elif mode == 'val':
            img_dir = os.path.join(new_data_dir, 'images', mode)
            os.makedirs(img_dir, exist_ok=True)

            label_dir = os.path.join(new_data_dir, 'labels', mode)
            os.makedirs(label_dir, exist_ok=True)

        return img_dir, label_dir

    def convert(self):
        annotations = pd.read_csv(self.org_ann_pth)
        participants = os.listdir(self.org_dataset_dir)
        out_img_dir, out_label_dir = self.make_dirs(self.out_dir, self.mode)
        
        for p in participants:
            if p == '.DS_Store':  # For Mac
                continue
            part_subset_dir = os.path.join(self.org_dataset_dir, p)
            video_ids = os.listdir(part_subset_dir)
            for id in video_ids:
                id = id.strip()
                if id == '.DS_Store':
                    continue
                img_mask = []
                id_img_dir = os.path.join(self.org_dataset_dir, p, id)
                img_names = os.listdir(id_img_dir)
                for img_name in tqdm(img_names, desc=f"transform labels from {p, id}"):
                    out_label_path = os.path.join(out_label_dir, p + '_' + id + '-' + img_name.replace('.jpg', '.txt'))
                    if os.path.exists(out_label_path):
                        if img_name not in img_mask:
                            img_mask.append(img_name)
                        continue

                    bboxes_info = []
                    video_frame = int(img_name.split('.')[0]) - 0
                    ann_indices = annotations[(annotations['participant_id'] == p) & (annotations['video_id'] == id) &
                                              (annotations['frame'] == video_frame)].index.tolist()
                    img = cv2.imread(os.path.join(id_img_dir, img_name))

                    for ann_idx in ann_indices:
                        if annotations.loc[ann_idx].bounding_boxes == '[]':  # without object
                            continue
                        else:
                            if img_name not in img_mask:
                                img_mask.append(img_name)
                            bboxes = eval(annotations.loc[ann_idx].bounding_boxes)  # (y1, x1, h, w)
                            for bbox in bboxes:
                                bbox = self.coor_transform(bbox, img.shape)  # [x_c, y_c, w, h]
                                bbox.insert(0, annotations.loc[ann_idx]['noun_class'])
                                bboxes_info.append(bbox)

                    if len(bboxes_info) > 1:  # for one object, there are more than one bounding box
                        temp = bboxes_info[:]
                        for i in range(len(bboxes_info) - 1):
                            for box_info in bboxes_info[i + 1:]:
                                iou = self.bbox_iou(bboxes_info[i][1:], box_info[1:])
                                if iou > self.iou_thres:
                                    if box_info in temp:
                                        del temp[temp.index(box_info)]

                        bboxes_info = temp[:]
                    self.write_to_file(out_label_path, bboxes_info)

                self.move_images(out_img_dir, id_img_dir, img_mask, p + '_' + id)


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--org_ann_path', type=str, required=True,
                       help='the local path to epic-55 annotations')
    parse.add_argument('--org_names_path', type=str, required=True,
                       help='the local path to epic-55 noun classes')
    parse.add_argument('--org_dataset_dir', type=str, required=True,
                       help='the local directory of original dataset')
    parse.add_argument('--out_dir', type=str, required=True,
                       help='the output directory')
    parse.add_argument('--mode', type=str, required=True, choices=['train', 'val'], default='train', 
                       help='the train dir or val dir')
    parse.add_argument('--iou_thres', type=float, required=True, default=0.4, help='to eliminate duplicate bounding boxes')
    parse.add_argument('--write_flag', type=bool, default=False, help='whether to write data configuration files')
    opt = parse.parse_args()

    epic2darknet = Epic2Darknet(opt.org_ann_path, opt.org_names_path,
                                opt.org_dataset_dir, opt.out_dir, opt.mode, opt.iou_thres)
    epic2darknet.convert()

    if opt.write_flag:
        modes = ['train', 'val']
        try:
            epic2darknet.add_extra_files(opt.output_dir)
        except FileNotFoundError:
            modes.remove(opt.mode)
            epic2darknet.make_dirs(opt.output_dir, modes[-1])
            epic2darknet.add_extra_files(opt.output_dir)
