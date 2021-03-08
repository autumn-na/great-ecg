# Using Conditional Generative Models to Protect Patient's Privacy

## Introduction

### Current
ECG Signal analysis is essential for medical development.   
However, using ECG signal as it is for neural network learning cause leaking patient information.   

### Project Topic
Using conditional generative models to protect patient’s privacy

### Project Goal
To convert the ECG Signal so that the machine does not know whose signal it is.

### Code Description
It was programmed as python, and Google Colab. 
MinhyoungNa_Classifier_Update_022421_copy.ipynb
* Implementation of Classifier   

MinhyoungNa_VAE_Poisoning_Update_022421.ipynb
* Implementation of VAE, which modifies ECG Signal   

## Preliminary

### Variational Auto Encoder (VAE)
![image](https://user-images.githubusercontent.com/54922741/110275725-f8fd8e80-8014-11eb-9e33-603626048938.png)   
The dimension is compressed as the data passes through the encoder.   
And returns to the original dimension space when it passes through the decoder.   
The output data may not be exactly the same as the input data.   

### ECG Signal
![image](https://user-images.githubusercontent.com/54922741/110275848-36621c00-8015-11eb-94be-bb43ebb0a2ee.png)   
Health care community analyzes the period of ECG Signals for disease detection,   
which is measured heart rate of the patient.

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

```python
QRS_WIDTH = 50
NON_QRS_WIDTH = 50

data_valid = np.zeros((0, 200))
labels_valid = []
data_qrs = np.zeros((0, 50))
data_non_qrs = np.zeros((0, 50))

cnt = -1
for item in data[:data_using]:
  cnt += 1
  peak = np.argmax(item)
  if peak < QRS_WIDTH or peak > dataLen-QRS_WIDTH or peak < QRS_WIDTH/2 + NON_QRS_WIDTH or peak > dataLen - (QRS_WIDTH/2 + NON_QRS_WIDTH):  #invailed data
    continue
  data_qrs = np.vstack([data_qrs, item[peak-(QRS_WIDTH//2):peak+(QRS_WIDTH//2)]])
  data_non_qrs = np.vstack([data_non_qrs, item[peak-QRS_WIDTH//2-NON_QRS_WIDTH:peak-QRS_WIDTH//2]])
  data_valid = np.vstack([data_valid, item])
  labels_valid.append(labels[cnt])

data_non_qrs = data_non_qrs.astype(np.float32)
data_qrs = data_qrs.astype(np.float32)
```

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

```python
class Classifier(nn.Module):
  def __init__(self):
    super(Classifier, self).__init__()

    self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=6) #195
    self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=6) #195
    self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=6) #60
    self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6) #60
    self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6) #60
    self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6) #15
    self.fc7 = nn.Linear(6*128 , 128)
    self.fc8 = nn.Linear(128, len(classes))
    self.dropout = nn.Dropout(p=0.25)

  def forward(self, x):
    h = F.relu(self.conv1(x.unsqueeze(1))) #195
    h = F.max_pool1d(F.relu(self.conv2(h)), 2) #190, 95
    h = F.relu(self.conv3(h)) #90
    h = F.max_pool1d(F.relu(self.conv4(h)), 3) #85, 28
    h = F.relu(self.conv5(h)) #23
    h = F.max_pool1d(F.relu(self.conv6(h)), 3) #18, 6
    h = h.view(-1, 6*128)
    h = self.dropout(self.fc7(h))
    h = self.dropout(self.fc8(h))
    return h
```

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

```python
class VAE(nn.Module):
  def __init__(self):
    super(VAE, self).__init__()
    
    Z_DIM = 5

    # encoder part
    self.fc1 = nn.Linear(50, 40)
    self.fc2 = nn.Linear(40, 40)
    self.fc3 = nn.Linear(40, 30)
    self.fc4 = nn.Linear(30, 30)
    self.fc5_m = nn.Linear(30, Z_DIM)
    self.fc5_l = nn.Linear(30, Z_DIM)

    # decoder part
    self.fc6 = nn.Linear(Z_DIM, 30)
    self.fc7 = nn.Linear(30, 30)
    self.fc8 = nn.Linear(30, 40)
    self.fc9 = nn.Linear(40, 40)
    self.fc10 = nn.Linear(40, 50)

  def encoder(self, x):
    h = nn.functional.relu(self.fc1(x))
    h = nn.functional.relu(self.fc2(h))
    h = nn.functional.relu(self.fc3(h))
    h = nn.functional.relu(self.fc4(h))
    return self.fc5_m(h), self.fc5_l(h) # mu, log_var

  def sampling(self, mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu) # return z sample

  def decoder(self, z):
    h = nn.functional.relu(self.fc6(z))
    h = nn.functional.relu(self.fc7(h))
    h = nn.functional.relu(self.fc8(h))
    h = nn.functional.relu(self.fc9(h))
    return torch.sigmoid(self.fc10(h))

  def forward(self, x):
      mu, log_var = self.encoder(x.view(-1, 50))
      z = self.sampling(mu, log_var)
      return self.decoder(z), mu, log_var
```

## Results
### Poisoning
![image](https://user-images.githubusercontent.com/54922741/110277695-25b3a500-8019-11eb-8cac-f0c2476463ee.png)   

###Classifying
![image](https://user-images.githubusercontent.com/54922741/110277752-47ad2780-8019-11eb-8b3e-02418eb4af06.png)      
I reduced accuracy of classifying ECG Data 98.34% to 3.5%.
