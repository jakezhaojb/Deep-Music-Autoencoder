Stacked Auto-encoder
===

Introduction
---
Following the hot trend of Deep Learning, this repo implements the Stacked Auto-encoders which involves a whole bunch of configuration choices such as 'Denoising', 'Fine-tuning' and 'Dropout'. Stacked Auto-encoder could be used in many kinds of cases. If you do not enable fine-tuning, Stacked Auto-encoder could be treated as an unsupervised learning method; while if fine-tuning option is to be true, this tool is a supervised learning machine which adopts Softmax on top of the neural networks.

Above is a brief introduction to Stacked Auto-encoder. More: [http://deeplearning.net/](http://deeplearning.net/).

Installation
---
The two version of Auto-encoder - python version and C++ version.

Python version has a pre-requisite of [https://github.com/douban/dpark](Dpark), which is a Python clone of Spark. I have mainly adopted Dpark as map-reduce programming, substituting those simple loops, and thus get accelerated. 

Meanwhile, C++ version requires [http://paracel.io](paracel) as the parallel computing framework; paracel was produced by [http://douban.com](douban) following the idea of **Parameter Server** by Jeaf Dean @Google. Paracel could be easily applied in stochastic gradient descent alike training frameworks.

Besides this, if you run the program on a shared pool of servers, I recommend you to deploy [http://mesos.apache.org/](mesos) to manage the cluster.

Usage
----
To be updated.

Contact
---
zhaojunbo1992chasing@gmail.com
