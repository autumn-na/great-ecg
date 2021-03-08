# Using Conditional Generative Models to Protect Patient's Privacy

## Introduction

### Current
ECG Signal analysis is essential for medical development.   
However, using ECG signal as it is for neural network learning cause leaking patient information.   

### Project Topic
Using conditional generative models to protect patient’s privacy

### Project Goal
To convert the ECG Signal so that the machine does not know whose signal it is.

## Preliminary

### Variational Auto Encoder (VAE)
![image](https://user-images.githubusercontent.com/54922741/110275725-f8fd8e80-8014-11eb-9e33-603626048938.png)   
The dimension is compressed as the data passes through the encoder.   
And returns to the original dimension space when it passes through the decoder.   
The output data may not be exactly the same as the input data.   

### ECG Signal
![image](https://user-images.githubusercontent.com/54922741/110275848-36621c00-8015-11eb-94be-bb43ebb0a2ee.png)   
Health care community analyzes the period of ECG Signals for disease detection.   

### Mathematica lECG Signal Analysis Methods
![image](https://user-images.githubusercontent.com/54922741/110276045-b12b3700-8015-11eb-9488-a6cfcf9673d5.png)   
![image](https://user-images.githubusercontent.com/54922741/110276163-e6d02000-8015-11eb-8707-60face0a1c91.png)   

### Database
![image](https://user-images.githubusercontent.com/54922741/110276275-226aea00-8016-11eb-943f-57288a6509b8.png)   
![image](https://user-images.githubusercontent.com/54922741/110276328-3ca4c800-8016-11eb-8dfa-32d612cac9fc.png)   
MIT-BIT Arrhythmia Database   
The graph below shows sampled ECG data of one person displayed overlappingly.   


## Implementations

### Idea
![image](https://user-images.githubusercontent.com/54922741/110276708-13d10280-8017-11eb-8996-d6cd6bfbdb15.png)   
Modify data which is not critical in diagnosis.   
So I focused on QRS complex in the ECG data.
* QRS complex is usually checked up to diagnose arrhythmia.      
Non-QRS data are not important parts of the data.
* However, interval between peaks (RR interval) is still important.

### Non-QRS
![image](https://user-images.githubusercontent.com/54922741/110276749-2ea37700-8017-11eb-8ed2-dff3ebaed8d9.png)   
I found non-QRS data between peaks in ECG signal.
* To avoid QRS complex, I selected data which keep constant distance from the peak.

### Classifier
![image](https://user-images.githubusercontent.com/54922741/110277063-e5075c00-8017-11eb-87a2-003e55685c27.png)
<p> 
I judged that the data was well-modified if AI could not properly classify the poisoned ECG data which contains generated data.
* The classifier is trained from original data.

### Architecture - Classifier
![image](https://user-images.githubusercontent.com/54922741/110277238-344d8c80-8018-11eb-96a0-2738b8a55234.png)
Category | Description
---- | ----
Loss Function | Cross entropy
Optimizer | Adam
Learning Rate | 1e-5
Batch Size | 32
Epoches | 25

### VAE
![image](https://user-images.githubusercontent.com/54922741/110277104-ff413a00-8017-11eb-92ee-a920fdfa13c9.png)   
I generated non-QRS data with VAE.
* The VAE is trained from original non-QRS data.
* By adjusting the latent variable, the random data are generated.

### Architecture - VAE
![image](https://user-images.githubusercontent.com/54922741/110277580-e08f7300-8018-11eb-99fe-f8184e74546b.png)   
Category | Description
---- | ----
Loss Function | Binary cross entropy, KL-divergence
Optimizer | Adam
Learning Rate | 1e-3
Batch Size | 64
Epoches | 50


## Results
### Poisoning
![image](https://user-images.githubusercontent.com/54922741/110277695-25b3a500-8019-11eb-8cac-f0c2476463ee.png)   

###Classifying
![image](https://user-images.githubusercontent.com/54922741/110277752-47ad2780-8019-11eb-8b3e-02418eb4af06.png)   
I reduced accuracy of classifying ECG Data 98.34% to 3.5%
