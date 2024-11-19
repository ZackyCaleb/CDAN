# CDAN
The official code repository for the paper "Self-Supervised Facial Expression Parsing: Unveiling Global Patterns through Facial Action Units."


* First Step
  * Please set your environments following the requirements.


* Second Step
  * Please download related datasets from [Affectnet](http://mohammadmahoor.com/affectnet/), [Oulu-CASIA](https://www.oulu.fi/en), and [KDEF](http://www.emotionlab.se/kdef/).
  * Meanwhile, please use [Openface toolkit](https://github.com/TadasBaltrusaitis/OpenFace) to get the segment faces and facial action units of these images. Please construct a ```.pkl``` file with facial action units and place it in a folder, which will be used to train the CDAN.
  * Finally, allocate the training and testing datasets and place them under folder datasets.


* Third Step
  * An intuitive demo can be executed directly by  ```python main.py```.
  * If you utilize your datasets, you can adjust the parameters of ```opt.py``` and subsequently run ```main.py```.


* Acknowledgment \
 Appreciate the works of those who came before: \
 [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841) \
 [SARGAN: Spatial Attention-Based Residuals for Facial Expression Manipulation](https://ieeexplore.ieee.org/abstract/document/10065495)
