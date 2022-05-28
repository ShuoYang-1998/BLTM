# ICML 2022 Estimating Instance-dependent Bayes-label Transition Matrix using a Deep Neural Network

This is the code for the paper:
[Estimating Instance-dependent Bayes-label Transition Matrix using a Deep Neural Network](https://arxiv.org/pdf/2105.13001.pdf)      
Shuo Yang, Erkun Yang, Bo Han, Yang Liu, Min Xu, Gang Niu, Tongliang Liu.

### Install requirements.txt
~~~
pip install -r requirements.txt
~~~

## Experiments
We verify the effectiveness of the proposed method on synthetic noisy datasets. In this repository, we provide the used [datasets](https://drive.google.com/open?id=1Tz3W3JVYv2nu-mdM6x33KSnRIY1B7ygQ) (the images and labels have been processed to .npy format). You should put the datasets in the folder “data” when you have downloaded them.       
Here is a training example: 
```bash
python main.py \
    --dataset mnist \
    --noise_rate 0.2 \
    --gpu 0
```
If you find this code useful in your research, please cite  
```bash
@inproceedings{yang2022bltm,
  title={Estimating Instance-dependent Bayes-label Transition Matrix using a Deep Neural Network},
  author={Yang, Shuo and Yang, Erkun and Han, Bo and Liu, Yang and Xu, Min and Niu, Gang and Liu, Tongliang},
  booktitle={ICML},
  year={2022}
}
```  
