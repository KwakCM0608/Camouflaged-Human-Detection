---
license: cc-by-nc-4.0
task_categories:
- image-segmentation
pretty_name: MoCH
size_categories:
- n<1K
---
# MoCH: Moving Camouflaged Human Object Detection Dataset

## Dataset Summary

MoCH is a dataset of Video Camouflaged Object Detection specific for human. It was created to support research in video camouflaged object detection..

The dataset contains:
- 98 video sequences and 4,091 human annotated frames.
- It has sampled frames and corrsponding image format annotations.
- Raw annotations are included. 

## Supported Tasks

This dataset supports the following tasks:
- Video Camouflaged Object Detection
- Video Camouflaged Human Detection
- Camouflaged Object Detection

## Dataset Structure

### Data Fields

Each video sequence in the dataset contains the following folders:

- `images`: the sampled original frames.
- `gts`: the binary mask/segmentation images for target camouflaged objects.
- 'annotations': raw annotation in coco format.

### Data Splits

| Split  | # Videos |
|--------|-----------|
| Train  | 49  |
| Val    | 15 |
| Test   | 34   |

## Dataset Creation

### Source Data

The data was collected from Youtube, and the original souce links of the videos in this dataset are available in this repo.

### Annotations

The data was annotated by human annotators and annotations are also reviewd by human.

### Licensing

The dataset is licensed under CC BY-NC 4.0. See [LICENSE] for details.