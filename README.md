# dl-label-conversion

This Repo is used to convert the image annotation (object detection) format into darknet format. So far, it supports [supervise.ly](https://supervise.ly/) and [EPIC-KITCHENS-55-annotations](https://github.com/epic-kitchens/epic-kitchens-55-annotations#epic_train_object_labelscsv).

# Usage
1.Download the Repo.
```
git clone https://github.com/SuyueLiu/dl-label-conversion.git
```
2.Run the \*.py file in command line.

```
super2darknet.py [-h] [--org_dir ORG_DIR] [--output_dir OUTPUT_DIR]
                        [--mode MODE] [--write_flag WRITE_FLAG]
```

```
epic2darknet.py [-h] --org_ann_path ORG_ANN_PATH --org_names_path
                       ORG_NAMES_PATH --org_dataset_dir ORG_DATASET_DIR
                       --out_dir OUT_DIR --mode {train,val} --iou_thres
                       IOU_THRES [--write_flag WRITE_FLAG]
```

3.Verify new labels

```
verify_labels.py [-h] --img_path IMG_PATH --label_path LABEL_PATH
                        --names_path NAMES_PATH
``` 

# Examples
- [supervise.ly to darknet](https://github.com/SuyueLiu/dissertation-diary/blob/master/prepare_data.ipynb)

- [epic kitchens to darknet](https://github.com/SuyueLiu/dissertation-diary/blob/master/epic55_training.ipynb)
