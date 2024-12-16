# Step 1: Install Git LFS if not already installed
echo "Installing Git LFS..."
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs -y

# Step 2: Initialize Git LFS
echo "Initializing Git LFS..."
git lfs install

# Step 3: Clone the CLIP model repository
echo "Cloning the CLIP model repository..."
mkdir -p deps/
cd deps/
git clone https://huggingface.co/openai/clip-vit-large-patch14
cd ..

echo "Setup complete."