import streamlit as st
import torch
import numpy as np
from PIL import Image
from model import CVAE, LATENT_DIM, NUM_CLASSES # Import from model.py

# Set the device to CPU for deployment
device = torch.device('cpu')

@st.cache_resource
def load_model():
    """Load the trained CVAE model."""
    model = CVAE().to(device)
    # Load the state dictionary. map_location ensures it works on CPU.
    model.load_state_dict(torch.load('cvae_mnist.pth', map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

def to_one_hot(labels, num_classes=NUM_CLASSES):
    """Helper function to convert labels to one-hot vectors."""
    one_hot = torch.zeros(labels.size(0), num_classes)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot.to(device)

def generate_images(model, digit_to_generate, num_images=5):
    """Generate a specified number of images for a given digit."""
    images = []
    for _ in range(num_images):
        with torch.no_grad():
            # Create a random latent vector (noise)
            z = torch.randn(1, LATENT_DIM).to(device)
            
            # Create the label for the desired digit
            label = torch.tensor([digit_to_generate]).to(device)
            label_one_hot = to_one_hot(label)
            
            # Generate the image from the decoder
            generated_tensor = model.decode(z, label_one_hot).cpu()
            
            # Reshape and convert to a displayable format
            img_array = generated_tensor.view(28, 28).numpy()
            img_array = (img_array * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            images.append(pil_image)
    return images

# --- Streamlit App UI ---

st.set_page_config(layout="wide")

st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using a Conditional VAE model trained from scratch.")

# Load the model
model = load_model()

# --- User Input ---
st.markdown("---")
col1, col2 = st.columns([1, 3])

with col1:
    selected_digit = st.selectbox(
        "Choose a digit to generate (0-9):",
        options=list(range(10)),
        index=2 # Default to digit '2' as in the example
    )
    
    generate_button = st.button("Generate Images", type="primary")

# --- Image Display ---
if generate_button:
    with st.spinner(f"Generating 5 images of digit {selected_digit}..."):
        generated_images = generate_images(model, selected_digit, num_images=5)
    
    st.subheader(f"Generated images of digit {selected_digit}")
    
    # Display images in 5 columns
    cols = st.columns(5)
    for i, img in enumerate(generated_images):
        cols[i].image(
            img, 
            caption=f"Sample {i+1}", 
            width=150, # Control the display size
             use_container_width='auto'
        )
else:
    st.info("Select a digit and click 'Generate Images' to see the results.")