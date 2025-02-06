#!/bin/bash
set -e  # Exit on any error

# Configuration - replace these values
REPO_URL="https://github.com/piragi/rap.git"
META_URL="https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoicDR2YXBiaXU5Y2lodTAyeGFyZDA4OWtpIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczODgzNzYwN319fV19&Signature=GF2Yr%7EP3SEj%7E4dG8OYo0Hor7kNIXKowieSdkXgHR46MGt5dbKT3ffa53QklGYOkeCE7sGtMKXli8vYj5fNJHOvb6-h%7E0Z4AWOc%7ES-nDe6RLlviil6JqLt7H3ls4mPlKcxNjRUVZ1-zPXdrX8MgrM0AmH88dvrKN3th5NoiL5zZ9cEC6XR4Sga9Gbj%7E9Fmc5YTSkN3LM1nREmB5aFpUynRAOCarmLt-M1Vly5Ba32KEwlajPeUH9-YNun9Fbd2OhUpLzireIA%7Enuq31ZLfg7ZLxTx9uiadmje9CmbTuScSJhjvtMFDPBVssyhEZQV2jKoNepOAGcYJ19-aMCAAJKqwQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=9011235728967933"  # The temporary URL from Meta
ENV_NAME=".venv"  # Name of the virtual environment

echo "Starting setup process..."

# Install basic dependencies
echo "Installing base dependencies..."
apt-get update && apt-get install -y git wget python3-venv python3-pip

# Create and activate virtual environment
echo "Creating and activating virtual environment..."
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

