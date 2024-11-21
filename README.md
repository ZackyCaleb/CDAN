# CDAN ![image](https://github.com/user-attachments/assets/ff7f017a-1796-494a-88eb-46ed96c7befb)
The official code repository for the paper "Self-Supervised Facial Expression Parsing: Unveiling Global Patterns through Facial Action Units."
In this work, we introduce a novel self-supervised Codec Dual-Output Adversarial Network (CDAN), which is designed to parse facial expression features based on the Spatial Extend Attention (SEA) module and facial Action Units (AUs).
# Requirements and dependencies
 * Install PyTorch (version 1.10.0), and torchvision (version 0.11.0). Torch and torchvision are from http://pytorch.org.
 * Install requirements.txt (pip install -r requirements.txt)
 * Install OpenFace (version 2.2.0) from https://github.com/TadasBaltrusaitis/OpenFace.
 * Install [pytorch-fid](https://github.com/mseitzer/pytorch-fid) ```pip install torch-fidelity``` and [pytorch-fidelity](https://github.com/toshas/torch-fidelity) ``` pip install pytorch-fid```

# Data Preparation
   * Downloading the original images after obtaining official authorization for the mentioned datasets release: [Affectnet](http://mohammadmahoor.com/affectnet/), [Oulu-CASIA](https://www.oulu.fi/en), and [KDEF](http://www.emotionlab.se/kdef/).
   * Follow the official operation procedure of OpenFace to obtain segmented face regions and facial AUs.
   * Allocating training and testing datasets.
An example of this directory is shown in ```dataets/```.
To generate the affect_HaSa_au.pk, training datasets and testing datasets extract each image AU with OpenFace. A demo can be run: ```Python datasets_pre_col.py```.

# Run
## Training
* To Train and get checkpoints:
   ```
  Python main.py
   ```
* A checkpoint trained on AffectNet can be downloaded in [CDAN.pth](https://drive.google.com/file/d/1KuYYEIvtXQphuHlu9hpcbTe86kTXRbaE/view).

## Testing 
* To get reconstruct images:
   ```
  Python main.py  --mode rec --images_dir /datasets/imgs/testing
   ```

To quantitative evaluate reconstruct images:
   * evaluating IS score:
     ```
     fidelity --gpu 0 --isc --input1  path/to/reconstruct/imgs
     ```
   * evaluating FID score:
      ```
     python -m pytorch_fid /datasets/imgs/testing  path/to/reconstruct/imgs
      ```
   * evaluating ACD and ED scores:
      ```
     python ACD_ED.py
      ```
 
# Acknowledgment
 Appreciate the works and code repositories of those who came before: \
 [1] [Esser P, Rombach R, Ommer B. Taming transformers for high-resolution image synthesis[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 12873-12883.](https://arxiv.org/abs/2012.09841) \
 [2] [Akram A, Khan N. SARGAN: Spatial attention-based residuals for facial expression manipulation[J]. IEEE Transactions on Circuits and Systems for Video Technology, 2023, 33(10): 5433-5443.](https://ieeexplore.ieee.org/abstract/document/10065495)
