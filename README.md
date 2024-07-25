# HnN-Registration

Implement of "Head and Neck Registration"

## Overall Process
<img src = "https://github.com/user-attachments/assets/4a3c5cef-6a64-4cf5-a9a4-0aed3afa376f" width="100%" height="100%">

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
<img src = "https://github.com/user-attachments/assets/167f22c1-1fa5-4e8b-9a4d-4d42e001cd72" width="100%" height="100%">
<img src = "https://github.com/user-attachments/assets/b9ae4a99-2ed2-4dc4-b9e5-ddd70d8af3bc" width="100%" height="100%">

## Reference Codes
- [https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
- [https://github.com/zju3dv/LoFTR](https://github.com/zju3dv/LoFTR)
- [https://github.com/microsoft/CvT](https://github.com/microsoft/CvT)
