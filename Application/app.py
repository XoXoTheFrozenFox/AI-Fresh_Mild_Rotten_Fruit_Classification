import tkinter as tk
from tkinter import filedialog, Toplevel, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from pathlib import Path
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Define the class names (adjust as per your requirement)
class_names = {
    0: "Banana Good", 1: "Banana Rotten", 2: "Banana Mild",
    3: "Cucumber Good", 4: "Cucumber Mild", 5: "Cucumber Rotten",
    6: "Grape Good", 7: "Grape Mild", 8: "Grape Rotten",
    9: "Kaki Good", 10: "Kaki Mild", 11: "Kaki Rotten",
    12: "Papaya Good", 13: "Papaya Mild", 14: "Papaya Rotten",
    15: "Peach Good", 16: "Peach Mild", 17: "Peach Rotten",
    18: "Pear Good", 19: "Pear Mild", 20: "Pear Rotten",
    21: "Pepper Good", 22: "Pepper Mild", 23: "Pepper Rotten",
    24: "Strawberry Mild", 25: "Strawberry Rotten",
    26: "Tomato Good", 27: "Tomato Mild", 28: "Tomato Rotten",
    29: "Watermelon Good", 30: "Watermelon Mild", 31: "Watermelon Rotten"
}

# Define the transformation for preprocessing images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class FruitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry('400x600')
        self.root.title("Fruit Classifier")
        self.root.configure(bg='black')  # Set background color of the root window
        
        self.load_model()  # Load the PyTorch model
        self.vgg_model = VGG16(weights='imagenet', include_top=True)  # Load VGG16 model with pre-trained weights

        # Create UI elements with black background and white text
        self.label = tk.Label(root, text="Fruit Classifier App", font=("Helvetica", 16), bg='black', fg='white')
        self.label.pack(pady=10)
        
        self.canvas = tk.Canvas(root, width=300, height=300, bg='black', highlightthickness=0)
        self.canvas.pack(pady=10)
        
        # Create buttons with dark green background and bold text
        button_style = {'bg': 'dark green', 'fg': 'white', 'font': ('Helvetica', 12, 'bold')}

        self.label_prediction = tk.Label(root, text="", bg='black', fg='white')
        self.label_prediction.pack(pady=10) 

        self.btn_load_image = tk.Button(root, text="Load Image", command=self.load_image, **button_style)
        self.btn_load_image.pack(pady=10)
        
        self.btn_visualize_filters = tk.Button(root, text="Visualize Filters", command=self.visualize_filters, **button_style)
        self.btn_visualize_filters.pack(pady=10)
        
        self.btn_visualize_feature_maps = tk.Button(root, text="Visualize Feature Maps", command=self.visualize_feature_maps, **button_style)
        self.btn_visualize_feature_maps.pack(pady=10)
        
        self.popup = None  # Initialize popup variable

    def load_model(self):
        # Load the model from the specified path
        #C:/CODE(DO NOT DELETE PLS)/AI-Fresh_Mild_Rotten_Fruit_Classification/Model/fruit_classifier.pth
        #C:/CODE/Code/CODE ON GITHUB/AI-Fresh_Mild_Rotten_Fruit_Classification/Model/fruit_classifier.pth
        model_path = Path('C:/CODE/Code/CODE ON GITHUB/AI-Fresh_Mild_Rotten_Fruit_Classification/Model/fruit_classifier.pth')
        self.model = Net()
        # Load the model with map_location='cpu'
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        print("Model loaded successfully")

    def load_image(self):
        file_path = filedialog.askopenfilename(initialdir="/", title="Select an Image",
                                            filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if file_path:
            image = Image.open(file_path)
            image = image.resize((300, 300)) 
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            self.image_path = file_path  # Store the file path for later use
            self.predict_image()  # Predict the image right after loading it

    def predict_image(self):
        if not hasattr(self, 'image_path'):
            messagebox.showerror("Error", "No image loaded.")
            return
        
        # Load and preprocess the image
        img = Image.open(self.image_path).convert('RGB')  # Ensure image is in RGB format
        img = preprocess(img)  # Apply the preprocessing transformations
        
        # Add batch dimension: [1, 3, 224, 224]
        img_tensor = img.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            class_index = predicted.item()
        
        # Update the prediction label
        prediction = class_names.get(class_index, "Unknown Class")
        self.label_prediction.config(text=f"Prediction: {prediction}")

    def visualize_filters(self):
        # Extract filters from the first convolutional layer
        filters = self.vgg_model.get_layer('block1_conv1').get_weights()[0]
        filters = (filters - filters.min()) / (filters.max() - filters.min())
        n_filters = min(filters.shape[-1], 64)
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for i in range(n_filters):
            f = filters[:, :, :, i]
            for j in range(3):
                axes[i // 8, i % 8].imshow(f[:, :, j], cmap='gray')
                axes[i // 8, i % 8].axis('off')
        plt.show()

    def visualize_feature_maps(self):
        if not hasattr(self, 'image_path'):
            messagebox.showerror("Error", "Please load an image first.")
            return
        
        img = load_img(self.image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        # Redefine model to output from the last convolutional layer in each block
        ixs = [1, 3, 6, 8, 11]  # Indices of convolutional layers in VGG16
        outputs = [self.vgg_model.get_layer(self.vgg_model.layers[i].name).output for i in ixs]
        model = Model(inputs=self.vgg_model.inputs, outputs=outputs)
        
        feature_maps = model.predict(img)
        
        for fmap in feature_maps:
            square = int(np.sqrt(fmap.shape[-1]))  # Calculate the number of rows and columns
            fig, axes = plt.subplots(square, square, figsize=(8, 8))
            ix = 1
            for i in range(square):
                for j in range(square):
                    if ix <= fmap.shape[-1]:
                        axes[i, j].imshow(fmap[0, :, :, ix-1], cmap='gray')
                    axes[i, j].axis('off')
                    ix += 1
            plt.show()

# Define the neural network model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(512)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(512 * 14 * 14, 512)
        self.fc2 = torch.nn.Linear(512, len(class_names))  # Adjusted for number of classes
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.dropout(torch.nn.functional.max_pool2d(self.relu(self.bn1(self.conv1(x))), 2))
        x = self.dropout(torch.nn.functional.max_pool2d(self.relu(self.bn2(self.conv2(x))), 2))
        x = self.dropout(torch.nn.functional.max_pool2d(self.relu(self.bn3(self.conv3(x))), 2))
        x = self.dropout(torch.nn.functional.max_pool2d(self.relu(self.bn4(self.conv4(x))), 2))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitClassifierApp(root)
    root.mainloop()
