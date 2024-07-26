import tkinter as tk
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from pathlib import Path

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

# Define the tkinter application class
class FruitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Classifier")
        self.root.configure(bg='black')  # Set background color of the root window
        
        self.load_model()  # Load the PyTorch model
        
        # Create UI elements with black background and white text
        self.label = tk.Label(root, text="Fruit Classifier App", font=("Helvetica", 16), bg='black', fg='white')
        self.label.pack(pady=10)
        
        self.canvas = tk.Canvas(root, width=300, height=300, bg='black', highlightthickness=0)
        self.canvas.pack(pady=10)
        
        self.btn_load_image = tk.Button(root, text="Load Image", command=self.load_image, bg='white', fg='black')
        self.btn_load_image.pack(pady=10)
        
        self.label_prediction = tk.Label(root, text="", bg='black', fg='white')
        self.label_prediction.pack(pady=10)
        
        self.popup = None  # Initialize popup variable
        
    def load_model(self):
        # Load the model from the specified path
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
            
            image_tensor = preprocess(image)
            image_tensor = image_tensor.unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction = class_names[int(predicted)]
                self.label_prediction.config(text=f"Predicted: {prediction}")
                

            self.show_preprocessed_image(image_tensor, file_path)

    def show_preprocessed_image(self, image_tensor, file_path):
        # Close existing popup if it exists
        if self.popup:
            self.popup.destroy()
        
        # Create a new popup window
        self.popup = Toplevel(self.root)
        self.popup.title("Preprocessed Image")
        self.popup.configure(bg='black')
        
        # Convert tensor back to image for display
        image = transforms.functional.to_pil_image(image_tensor.squeeze())
        
        # Display the preprocessed image
        photo = ImageTk.PhotoImage(image)
        label_image = tk.Label(self.popup, image=photo, bg='black')
        label_image.image = photo
        label_image.pack(padx=20, pady=20)
        
        # Display file path as label
        label_filepath = tk.Label(self.popup, text=f"File: {file_path}", bg='black', fg='white')
        label_filepath.pack(padx=20, pady=5)
        
        # Close button for the popup window
        btn_close = tk.Button(self.popup, text="Close", command=self.popup.destroy, bg='white', fg='black')
        btn_close.pack(padx=20, pady=10)
        
        # Ensure the popup window stays open
        self.popup.mainloop()

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

# Create the tkinter application
root = tk.Tk()
app = FruitClassifierApp(root)
root.mainloop()
