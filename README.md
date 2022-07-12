<h1 align='center' style='margin: auto; text-align: center'>
    Intel OpenVino Toolkit Inference Engine
</h1>    

## Introduction
Training deep learning models is computationally intensive and usually requires high-end GPUs and CPUs. OpenVINO™ toolkit allows to optimizing workloads across Intel® hardware, maximizing performance. The Model Optimizer imports, converts, and optimizes models.

<img width="1440" alt="Example Post" src=https://www.intel.com/content/dam/www/central-libraries/us/en/images/2022-06/openvino-chart-rwd.jpg>

Complete workflow can be performed into following four steps:

* Train a CNN model
* Generate Tensorflow frozen model (.PB file)
* Get OpenVino optimized intermediate representation (IR) model
* Inference using OpenVino IR model

<img width="1440" alt="Example Post" src=https://www.intel.com/content/dam/develop/public/us/en/images/diagrams-infographics/diagram-using-ov-full-16x9.jpg.rendition.intel.web.978.550.jpg>

Here, we are using MNIST dataset on this exercise.
<img width="1440" alt="Example Post" src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png">

Check out my [blog](https://maulikpandya1.medium.com/intel-openvino-toolkit-inference-engine-part-1-2-2a87d8db2999) for detailed explanation.
