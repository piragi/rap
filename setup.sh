#!/bin/bash
set -e  # Exit on any error

# Configuration - replace these values
REPO_URL="https://github.com/piragi/rap.git"
META_URL="https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoianZyc3pkaW9zaXJ5anpyZHB2NHFiNGkyIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczNjUzMzQ0OH19fV19&Signature=plrv8cYZHMbatxrkyJeI8uzegKVBj6RvixOgxQidtlrr6kKpZj7zxhwFCmTk04PlmMgGZOHAwFQuTUsnPZ0YsE2hgQjLa4rc2Dv9ACagvgNpbhmSnLBE9bqj3452a6V-weNKSqdfsyGGiYn1rkqsbrsHr6N1cCUBf-OMMdRb4b3D%7EcKkbVM8dwsmUo2bmNJSYWHTYsli3Z8tkedEgkCti5XI9NbIN5EOcP1dYTRXVFJPoiboEAeuQfi2sbx2RX66UsuG1-Wtg3p9QG5X0vXKsl5yNe6-4ncDxpHvN5UKTmEDlbDGxldfvfrJGzthtMYLMnJl4YIwjttsax260IcKbw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=954820303247478"  # The temporary URL from Meta
ENV_NAME="llama_env"  # Name of the virtual environment

echo "Starting setup process..."

# Install basic dependencies
echo "Installing base dependencies..."
apt-get update && apt-get install -y git wget python3-venv python3-pip

# Create and activate virtual environment
echo "Creating and activating virtual environment..."
python3 -m venv ~/$ENV_NAME
source ~/$ENV_NAME/bin/activate

# Verify we're in the virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
echo "Error: Virtual environment activation failed"
exit 1
fi

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Upgrade pip and install basic tools in the virtual environment
echo "Upgrading pip and installing basic tools..."
pip install -U pip setuptools wheel

# Create necessary directories
echo "Creating directories..."
mkdir -p ~/.llama/checkpoints/Llama3.2-3B

# Install Python packages in the virtual environment
echo "Installing Python packages..."
pip install llama-stack
pip install datasets transformers torch

# Install project-specific requirements if they exist
if [ -f "requirements.txt" ]; then
echo "Installing project requirements..."
pip install -r requirements.txt
fi

# Download the LLaMA model
echo "Downloading LLaMA model..."
if [ -z "$META_URL" ]; then
echo "Error: META_URL environment variable not set"
exit 1
fi
llama model download --source meta --model-id Llama3.2-3B-Instruct --meta-url "$META_URL"

# Verify the model files exist
echo "Verifying model files..."
required_files=("consolidated.00.pth" "params.json" "tokenizer.model")
for file in "${required_files[@]}"; do
if [ ! -f ~/.llama/checkpoints/Llama3.2-3B/$file ]; then
echo "Error: Missing required model file: $file"
exit 1
fi
done

# Download GSM8K dataset (it will be cached)
echo "Preparing to use GSM8K dataset..."
python -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main')"

# Create activation script
echo "Creating environment activation script..."
cat > ~/activate_env.sh << EOL
#!/bin/bash
source ~/$ENV_NAME/bin/activate
cd $(pwd)
echo "Environment activated and working directory set to: \$(pwd)"
EOL
chmod +x ~/activate_env.sh

echo "Setup completed successfully!"
echo "Model path: ~/.llama/checkpoints/Llama3.2-3B"
echo "Working directory: $(pwd)"
echo "To activate the environment in new sessions, run: source ~/activate_env.sh"

