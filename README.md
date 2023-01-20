# fgsm_attack_simulation
The repository provides a demo code for implementing fast gradient sign method (FGSM) using PyTorch. The code borrows from (PyTorch Official Tutorial)[https://pytorch.org/tutorials/beginner/fgsm_tutorial.html] and the base paper (Explaining and Harnessing Adversarial Examples)[https://arxiv.org/abs/1412.6572].  

FGSM is considered to be an adversarial attack on deep learning models, synonymous with viruses and malwares for computers. Adversarial attacks modify an original image such that it is undetectable to human eye or compels the model to misclassify. FGSM adds a pixel-wide perturbation in a single step as discussed in the aforementioned paper. 

You can vary the epsilon value for visualizing the impact of attack on MNIST images.

The results from the code are shown below:

![image](https://user-images.githubusercontent.com/26203136/213671672-a6caec39-889e-45f7-a021-0b77c67a3c5b.png)


![image](https://user-images.githubusercontent.com/26203136/213671704-379e641d-4322-4675-aeea-03d48df429e8.png)
