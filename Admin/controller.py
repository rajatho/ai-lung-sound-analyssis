import os
import hashlib
import socket
import mysql.connector as mssql
import os, sys
from time import sleep
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import seaborn as sns
def getMachine_addr():
	os_type = sys.platform.lower()
	command = "wmic bios get serialnumber"
	return os.popen(command).read().replace("\n","").replace("	","").replace(" ","")

def getUUID_addr():
	os_type = sys.platform.lower()
	command = "wmic path win32_computersystemproduct get uuid"
	return os.popen(command).read().replace("\n","").replace("	","").replace(" ","")

def extract_command_result(key,string):
    substring = key
    index = string.find(substring)
    result = string[index + len(substring):]
    result = result.replace(" ","")
    result = result.replace("-","")
    return result

def get_ip_address_of_host():
    mySocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        mySocket.connect(('10.255.255.255', 1))
        myIPLAN = mySocket.getsockname()[0]
    except:
        myIPLAN = '127.0.0.1'
    finally:
        mySocket.close()
    return myIPLAN

def runCNN():
    global text
    global classifier
    text.delete('1.0', END)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[49] * 100
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (46, 46, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = 8, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=50, shuffle=True, verbose=2)
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[49] * 100


def save_model():

    global model
    model = '../model/model.json'
    if os.path.exists(model):
        return True
    else:
        return False

import matplotlib.pyplot as plt
import pickle


def graph():
    # Load the training history
    with open('../model/history.pckl', 'rb') as f:
        data = pickle.load(f)

    acc = data['accuracy']
    loss = data['loss']

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(acc, marker='o', color='green')
    plt.title('CNN Epoch-wise Accuracy')
    plt.savefig('../graphs/accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss, marker='o', color='blue')
    plt.title('CNN Epoch-wise Loss')
    plt.savefig('../graphs/loss.png', dpi=300, bbox_inches='tight')
    plt.close()



def train():
        from ultralytics import YOLO

        # Load a YOLOv8 model (choose from yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
        model = YOLO("../model/yolov8s.pt")  # Small model (you can replace with other versions)

        # Train the model
        model.train(
                data="../Dataset/data/data.yaml",  # ✅ Make sure the path to data.yaml is correct
                epochs=50,
                imgsz=640,
                batch=16,
                name="yolov8_custom",
                project="runs/train",  # Folder to save results
                patience=10,           # Stop early if no improvement
                val=True               # Evaluate on validation set
        )



 
def md5(input_string):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()
def key_validate(str):
    conn = mssql.connect(
        user='root', password='ROOT', host='localhost', database='intrusion'
        )
    cur = conn.cursor()
    private_key = extract_command_result("SerialNumber",getMachine_addr()) + extract_command_result("UUID",getUUID_addr())
    if private_key in str:
        cur.execute("select * from SOFTKEY where private_key = %s and public_key = %s",(md5(private_key),md5(extract_command_result(private_key,str))))
        data=cur.fetchone()
        if data:
            return True
        else:
            return False
    else:
        return False
