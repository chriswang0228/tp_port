# tp_port
## Introduction
The prediction process is as below:
![image](https://github.com/chriswang0228/tp_port/blob/main/src/image.png)

## Installation 
  
1. Clone the repository:
```  
git clone https://github.com/chriswang0228/tp_port.git
```  
2. Navigate to the project directory:
```  
cd tp_port
``` 
3. Install the required packages:
```  
pip install -r requirements.txt
``` 
4. Run the download script:
```  
bash ./download.sh
```
## Usage

1. Run Inference:

To perform inference on the input images:
```  
bash ./infer.sh
```  
By default, the script will look for ``` .JPG ```  images in the specified folder.

2. If your script requires customization (e.g., changing clip paths or input data), edit the infer.sh file accordingly.