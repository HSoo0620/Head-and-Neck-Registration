# HnN-Registration

Implement of "Head and Neck Registration"

## Overall Process
<img src = "https://github.com/user-attachments/assets/303f52c1-57b3-433b-84f0-14733f7f7685" width="100%" height="100%">

## Prerequisites
- [python==3.8.8](https://www.python.org/)  <br/>
- [pytorch==1.8.1](https://pytorch.org/get-started/locally/)

## Installation
The required packages are located in ```requirements```.

    pip install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
    pip install -r requirement.txt

## Dataset
Segrap 2023 Datasets Link : https://segrap2023.grand-challenge.org/ <br/>

45 OARs segmentation from no-contrast and contrast-enhanced CT scans: Brain, Brainstem, Chiasm, Cochlea left, Cochlea right, Esophagus, Eustachian tube left, Eustachian tube right, Eye left, Eye right, Hippocampus left, Hippocampus right, Internal auditory canal left, Internal auditory canal right, Larynx, Larynx glottic, Larynx supraglottic, Lens left, Lens right, Mandible left, Mandible right, Mastoid left, Mastoid right, Middle Ear left, Middle ear right, Optic nerve left, Optic nerve right, Oral cavity, Parotid left, Parotid right, Pharyngeal constrictor muscle, Pituitary, Spinal cord, Submandibular left, Submandibular right, Temporal lobe left, Temporal lobe right, Thyroid, Temporomandibular joint left, Temporomandibular joint right, Trachea, Tympanic cavity left, Tympanic cavity right, Vestibular semicircular canal left, Vestibular semicircular canal right.

## Training
- Before training, pre-processing and initial affine registration must be performed.
  - For pre-processing, reference ```preprocessing_Segrap.py```.
  - For initial affine registration, reference ```Train_Affine.py```. <br/>
```python
python Train.py \
    --affine_model experiments/affine \
    --dataset_dir Dataset/Segrap_2023 \
    --save_validation_img True \
    --max_epoch 100 \
```
## Inference
```python
python Inference.py --affine_model experiments/affine --dataset_dir Dataset/Segrap_2023
```

## Main results
<img src = "https://github.com/user-attachments/assets/68e946ae-4d09-4962-9a5a-a2f32eff78b0" width="100%" height="100%">
<img src = "https://github.com/user-attachments/assets/0cf809ed-9e52-458b-9487-0f7a499734c5" width="100%" height="100%">

## Reference Codes
- [https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
- [https://github.com/zju3dv/LoFTR](https://github.com/zju3dv/LoFTR)
- [https://github.com/microsoft/CvT](https://github.com/microsoft/CvT)
