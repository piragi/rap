#!/bin/bash
set -e  # Exit on any error

# Configuration - replace these values
REPO_URL="https://github.com/piragi/rap.git"
META_URL="https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiMHp3azd2bGM2a3oyMWVkMjdmdThxemY4IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczNDQ0OTU0MX19fV19&Signature=kVqS9QpY4hSWki4K-zKTz6auwutUIQ0Nr6gVUGUcwbzMywA8JiYF%7E3WHEXAJvRAbWv6oVuY9HD7O%7E4A9NuOTfYfzZgf0NrhYENa0loz0noE3%7EdIsGT7x5f1COd9inbrLCp0CmeDya%7EbdVGF4qv6tsOBFLL3vdKpQVGITHbo9FMvle38zftENynnLH5WIW0WuCAMC%7Ei5z-QbSjPqqvPWnnwu%7EmxIq494tL-icKkVa5IrXortes6bcY7P6A%7EDiV0zAMRcQBuwrG4swaWwcOJa7fW7STadfWy9B4QceV-5lyu8bynfe0tM3TCl3PeyVSHTIu7ObSX%7EEV7w72Wvo2tP0%7Ew__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1718496768722423"  # The temporary URL from Meta
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

# Clone repository
echo "Cloning repository..."
git clone $REPO_URL
cd $(basename $REPO_URL .git)

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
llama model download --source meta --model-id Llama3.2-3B --meta-url "$META_URL"

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

