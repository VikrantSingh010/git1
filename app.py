from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import os

# Your existing code snippets here
import numpy as np
from PIL import Image

# Function to load images from file-paths
def load_imgs(file_paths, slice_, color, resize):
    default_slice = (slice(0, 250), slice(0, 250))  # Setting the default slice to the size of original dataset i.e., 250x250

    if slice_ is None: slice_ = default_slice
    else: slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))

    # Obtain the height and width of the image from slice
    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    # Resizing the image
    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)
        
    # Setting the dimensions for each image
    n_faces = len(file_paths)
    if not color: faces = np.zeros((n_faces, h, w), dtype=np.float32)
    else: faces = np.zeros((n_faces, h, w, 3), dtype=np.float32)

    # Loading images
    for i, file_path in enumerate(file_paths):
        pil_img = Image.open(file_path)
        pil_img = pil_img.crop((w_slice.start, h_slice.start, w_slice.stop, h_slice.stop))

        if resize is not None: pil_img = pil_img.resize((w, h))
        face = np.asarray(pil_img, dtype=np.float32)

        face /= 255.0
        if not color: face = face.mean(axis=2)
        faces[i, ...] = face

    return faces
from os import listdir
from os.path import join, isdir

# Function to fetch and load images from a certain directory
def fetch_lfw_deep_people(data_folder_path, slice_=None, color=False, resize=None, min_faces_per_person=0):
    person_names, file_paths = [], []

    # Fetching the names of the people and file paths
    for person_name in sorted(listdir(data_folder_path)):
        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path): continue

        paths = [join(folder_path, f) for f in sorted(listdir(folder_path))]
        n_pictures = len(paths)
        if n_pictures >= min_faces_per_person:
            person_name = person_name.replace("_", " ")
            person_names.extend([person_name] * n_pictures)
            file_paths.extend(paths)

    n_faces = len(file_paths)
    if n_faces == 0: raise ValueError("min_faces_per_person=%d is too restrictive" % min_faces_per_person)

    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)
    file_paths = np.array(file_paths)

    # Loading the images of the found file-paths
    faces = load_imgs(file_paths, slice_, color, resize)

    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    faces, target, paths = faces[indices], target[indices], file_paths[indices]
    return faces, target, target_names, paths
from skimage.feature import hog
from skimage.io import imread

# Function to compute HoG features (already provided)
def compute_hog(img):
  fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
  return fd, hog_image
import cv2
import numpy as np

# Functions to calculate LBP features (already provided)
def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val


def calcLBP(image):
    img = imread(image)
    height, width, channel = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
    return hist_lbp
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Load pre-trained ResNet-50 model
resnet = models.resnet50(pretrained=True)
# Remove the last fully connected layer
resnet = nn.Sequential(*list(resnet.children())[:-1])
# Set the model to evaluation mode
resnet.eval()

# Define a function to extract features from an image
def extract_cnn_features(image_path, model):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    # Add batch dimension
    image = image.unsqueeze(0)
    # Extract features
    with torch.no_grad():
        features = model(image)
    # Remove the batch dimension
    features = features.squeeze(0)
    return features
import matplotlib.pyplot as plt
DATA_DIR = "Dataset\lfw-deepfunneled\lfw-deepfunneled"   # Directory containing all the images

# Calling the function to fetch and load images
faces, target, target_names, paths = fetch_lfw_deep_people(DATA_DIR, resize=0.4, min_faces_per_person=40)

print(faces.shape, target.shape, target_names.shape)
h = faces.shape[1]
w = faces.shape[2]

# Plotting the faces
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(faces[i])
    plt.title(target_names[target[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()
import os

if os.path.exists('X_hog.npy') and os.path.exists('I_hog.npy'):
    # Load array if file exists
    X_hog = np.load('X_hog.npy')
    I_hog = np.load('I_hog.npy')
    print("HoG features loaded from files")
else:
    X_hog = []
    I_hog = []

    for i, face in enumerate(faces):
        hog_f, hog_i = compute_hog(face)
        X_hog.append(hog_f)
        I_hog.append(hog_i)
        if(i%100 == 0): print(f"Images Processed: [{i}/{paths.shape[0]}]")

    X_hog = np.array(X_hog)
    I_hog = np.array(I_hog)
    np.save('X_hog.npy', X_hog)
    np.save('I_hog.npy', I_hog)
    # Plotting HoG features
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(I_hog[i])
    plt.title(target_names[target[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()
if os.path.exists('X_lbp.npy'):
    # Load array if file exists
    X_lbp = np.load('X_lbp.npy')
    print("LBP features loaded from file X_lbp.npy")
else:
    X_lbp = []

    for i, path in enumerate(paths):
        lbp = calcLBP(path)
        if(i%100 == 0): print(f"Images Processed: [{i}/{paths.shape[0]}]")

    X_lbp = np.array(X_hog)
    np.save('X_lbp.npy', X_lbp)
    # Plotting the histograms for LBP Features
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.plot(X_lbp[i])
    plt.title(target_names[target[i]])
    plt.xlabel("Pixel Value")
plt.tight_layout()
plt.show()
if os.path.exists('X_cnn.npy'):
    X_cnn = np.load('X_cnn.npy')
    print("CNN features loaded from X_cnn.npy")
else:
    X_cnn = []

    for i, path in enumerate(paths):
        cnn = extract_cnn_features(path, resnet)
        X_cnn.append(cnn)
        if(i%100 == 0): print(f"Images Processed: [{i}/{paths.shape[0]}]")

    # Stack the features into a single numpy array
    X_cnn = torch.stack(X_cnn).numpy()

    np.save('X_cnn.npy', X_cnn)
    from sklearn.model_selection import train_test_split

# Storing faces as 1-D array
X = faces.reshape(len(faces), -1)
y = target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Getting the train and test indices
train_indices = np.arange(len(X))[~np.isin(np.arange(len(X)), np.arange(len(X_test)))]
test_indices = np.arange(len(X))[np.isin(np.arange(len(X)), np.arange(len(X_test)))]
from sklearn.decomposition import PCA

# Fitting data to PCA model
pca = PCA()
pca.fit(X_train)

# Cummulative Variance Ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
# Setting the total variance and n_components
target_variance = 0.98
n_comp = np.argmax(cumulative_variance_ratio >= target_variance) + 1

print(f"n_components = {n_comp}")

# Plotting the cummulative variance ratio with n_components
plt.plot(cumulative_variance_ratio)
plt.axvline(x=n_comp, color='red', linestyle='--', label='n_components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()
# Calling another instance of PCA with n_components = n_com
pca = PCA(n_comp)
pca.fit(X_train)

# Transforming the dataset to n_comp dimensions
X_train_t = pca.transform(X_train)
X_test_t = pca.transform(X_test)

fig, axes = plt.subplots(3, 4, figsize=(6, 6))

# plotting first 12 eigenfaces
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape((h, w)), cmap='gray')
    ax.set_title(f"Eigenface {i+1}")
    ax.axis('off')

plt.tight_layout()
plt.show()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Fitting the dataset to LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_t, y_train)

# Transforming the dataset along the LDA projection vector
X_train_t = lda.transform(X_train_t)
X_test_t = lda.transform(X_test_t)
# Writing the transformed dataset into csv files for easy accessibility
with open("./test.csv", "w") as f:
    for i in range(X_test_t.shape[0]):
        for j in range(X_test_t.shape[1]):
            f.write(f"{X_test_t[i][j]},")
        f.write(f"{y_test[i]}\n")

with open("./train.csv", "w") as f:
    for i in range(X_train_t.shape[0]):
        for j in range(X_train_t.shape[1]):
            f.write(f"{X_train_t[i][j]},")
        f.write(f"{y_train[i]}\n")

print(f"Dimensions of transformed data: {X_train_t.shape[1]}")
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Reading the test and train files to load the transformed dataset in form of DataFrames
df = pd.read_csv("./train.csv", header=None)
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, -1]

df = pd.read_csv("./test.csv", header=None)
X_test = df.iloc[:, :-1]
y_test = df.iloc[:, -1]

# Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"Accuracy for Naive Bayes Classifier: {np.mean(gnb.predict(X_test) == y_test)*100:.2f}%")

# K-Nearest neighbors classifier with k = 5
knn = KNeighborsClassifier(5)
knn.fit(X_train, y_train)
print(f"Accuracy for K-Nearest Neighbor Classifier: {np.mean(knn.predict(X_test) == y_test)*100:.2f}%")

# Random Forets Classifier
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
print(f"Accuracy for Random Forest Classifier: {np.mean(forest.predict(X_test) == y_test)*100:.2f}%")

# Support Vector Classifier with linear kernel
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
print(f"Accuracy for Support Vector Classifier with linear kernel = {np.mean(svm.predict(X_test) == y_test)*100:.2f}%")
import torch

# Converting the train and test data to tensors
X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.int64))
X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.int64))
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# ANN class
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        # defining the sequence of layers
        self.layers = nn.Sequential(
            nn.Linear(X_train.shape[1], 200),
            nn.ReLU(),  # ReLU activation in each hidden layer
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, y_train.unique().shape[0])
        )
    # Forwarding the input to the layers
    def forward(self, x):
        logits = self.layers(x)
        return logits
    
# Defining train dataset and train loader for batch size of 16
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Calling the model
model = ANN()
criterion = nn.CrossEntropyLoss()   # Cross Entropy Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)    # Adam optimizer

# training the model for 20 epochs
for epoch in range(20):
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{20}], Loss: {loss.item()}")

    # Indentation corrected here
    with torch.no_grad():
        model.eval()
        outputs = model(X_test_tensor)  # Predictions for test data
        probabilities = nn.functional.softmax(outputs, dim=1)   # Applying softmax activation for output layer
        _, predicted = torch.max(probabilities, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)   # Computing accuracy of the model
        print(f"Accuracy for Artificial Neural Network with Cross Entropy Loss as Loss function = {accuracy*100:.2f}%")
    
    img_path = input("Provide path to the image: ") 


print(img_path)

pil_img = Image.open(img_path)
pil_img = pil_img.resize((100, 100))
face = np.asarray(pil_img, dtype=np.float32)

face /= 255.0
face = face.mean(axis=2)

face = face.reshape(-1)

face_pca = pca.transform([face])
face_lda = lda.transform(face_pca)

print(f"Predicted: {target_names[gnb.predict(face_lda)][0]}")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_path = request.form['img_path']
    pil_img = Image.open(img_path)
    pil_img = pil_img.resize((100, 100))
    face = np.asarray(pil_img, dtype=np.float32)
    face /= 255.0
    face = face.mean(axis=2)
    face = face.reshape(-1)
    face_pca = pca.transform([face])
    face_lda = lda.transform(face_pca)
    prediction = target_names[gnb.predict(face_lda)][0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
