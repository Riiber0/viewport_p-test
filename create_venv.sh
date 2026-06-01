#!/bin/bash

# Script to create virtual environment with Python 3.8+ and install dependencies
# Specifically configured for Keras 2.13.1 and TensorFlow
# Author: Assistant
# Date: $(date +%Y-%m-%d)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Setting up Python Environment with Keras 2.13.1${NC}"
echo -e "${GREEN}================================${NC}"

# Check Python version (3.8+ is sufficient for Keras 2.13.1)
echo -e "${YELLOW}Checking Python version...${NC}"
if command -v python3.8 &> /dev/null; then
    PYTHON_CMD="python3.8"
    echo -e "${GREEN}✓ Python 3.8 found${NC}"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
    echo -e "${GREEN}✓ Python 3.9 found${NC}"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo -e "${GREEN}✓ Python 3.10 found${NC}"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo -e "${GREEN}✓ Python 3.11 found${NC}"
else
    echo -e "${RED}Python 3.8+ not found!${NC}"
    echo -e "${YELLOW}Attempting to install Python 3.8...${NC}"
    
    # Detect operating system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux (Ubuntu/Debian)
        sudo apt update
        sudo apt install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt update
        sudo apt install -y python3.8 python3.8-venv python3.8-dev
        PYTHON_CMD="python3.8"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install python@3.8
        PYTHON_CMD="python3.8"
    else
        echo -e "${RED}Please install Python 3.8+ manually${NC}"
        exit 1
    fi
fi

# Verify Python installation
$PYTHON_CMD --version
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to find/install Python${NC}"
    exit 1
fi

# Create project directory (if it doesn't exist)
PROJECT_DIR="$(pwd)"
VENV_DIR="$PROJECT_DIR/venv_keras"

echo -e "${YELLOW}Creating virtual environment at: $VENV_DIR${NC}"

# Remove existing virtual environment (if exists)
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Existing virtual environment found. Removing...${NC}"
    rm -rf "$VENV_DIR"
fi

# Create new virtual environment
$PYTHON_CMD -m venv "$VENV_DIR"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Virtual environment created successfully${NC}"
else
    echo -e "${RED}Failed to create virtual environment${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment${NC}"
    exit 1
fi

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install specific versions for compatibility with Keras 2.13.1
echo -e "${YELLOW}Installing dependencies (Keras 2.13.1 + TensorFlow)...${NC}"
echo -e "${GREEN}This may take several minutes...${NC}"

# Create requirements.txt with specific versions
cat > requirements.txt << EOF
# Core dependencies for Keras 2.13.1
numpy>=1.21.0,<1.24.0
pandas>=1.3.0
tensorflow==2.13.0
keras==2.13.1

# Scikit-learn and preprocessing
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.4.0

# Scientific computing
scipy>=1.7.0

# Progress bar
progressbar2>=4.0.0

# Optional but recommended for TensorFlow
protobuf>=3.20.0,<3.21.0
h5py>=3.7.0
EOF

# Install all requirements
pip install -r requirements.txt

# Verify specific Keras version
echo -e "${YELLOW}Verifying installations...${NC}"
python -c "
try:
    import keras
    print(f'✓ Keras version: {keras.__version__}')
    assert keras.__version__ == '2.13.1', f'Expected 2.13.1, got {keras.__version__}'
except Exception as e:
    print(f'✗ Keras verification failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Keras 2.13.1 installed correctly${NC}"
else
    echo -e "${RED}⚠ Keras version mismatch. Reinstalling...${NC}"
    pip uninstall keras tensorflow -y
    pip install tensorflow==2.13.0 keras==2.13.1
fi

# Check for custom modules
echo -e "${YELLOW}Checking for custom modules...${NC}"

# Check if utils exists
if [ ! -f "utils.py" ]; then
    echo -e "${RED}Warning: utils.py not found in current directory${NC}"
    echo -e "${YELLOW}Please create this file with required functions${NC}"
fi

# Check if model.py exists
if [ ! -f "model.py" ]; then
    echo -e "${RED}Warning: model.py not found in current directory${NC}"
    echo -e "${YELLOW}Please create this file with CNN_GRU, baseModel, CNN_LSTM classes${NC}"
fi

# Check if graph_navagation.py exists
if [ ! -f "graph_navagation.py" ]; then
    echo -e "${RED}Warning: graph_navagation.py not found in current directory${NC}"
    echo -e "${YELLOW}Please create this file with NavigationGraphPredictor class${NC}"
fi

# Final information
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}✓ Setup completed successfully!${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "${YELLOW}Virtual environment location:${NC} $VENV_DIR"
echo -e "${YELLOW}Python version:${NC} $($PYTHON_CMD --version)"
echo -e ""
echo -e "${YELLOW}To activate the virtual environment:${NC}"
echo -e "source $VENV_DIR/bin/activate"
echo -e ""
echo -e "${YELLOW}To deactivate the virtual environment:${NC}"
echo -e "deactivate"
echo -e ""
echo -e "${YELLOW}To run your Python script:${NC}"
echo -e "source $VENV_DIR/bin/activate && python your_script.py"
echo -e ""
echo -e "${YELLOW}Installed packages:${NC}"
pip list | grep -E "numpy|pandas|tensorflow|keras|scikit-learn|matplotlib|scipy|progressbar"

# Create quick activation script
cat > activate_keras_env.sh << EOF
#!/bin/bash
source $VENV_DIR/bin/activate
echo "✓ Keras virtual environment activated!"
echo "Python: \$(python --version)"
echo "Keras: \$(python -c 'import keras; print(keras.__version__)')"
echo "TensorFlow: \$(python -c 'import tensorflow as tf; print(tf.__version__)')"
EOF

chmod +x activate_keras_env.sh
echo -e "${GREEN}✓ Quick activation script created: ./activate_keras_env.sh${NC}"

# Comprehensive test of all imports
echo -e "${YELLOW}Testing all imports with Keras 2.13.1...${NC}"
python -c "
import sys
import os

# Set environment variables as in your code
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_NUM_INTEROP_THREADS'] = '2' 
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

try:
    print('Testing imports...')
    import numpy as np
    print(f'  ✓ NumPy: {np.__version__}')
    
    import pandas as pd
    print(f'  ✓ Pandas: {pd.__version__}')
    
    import tensorflow as tf
    print(f'  ✓ TensorFlow: {tf.__version__}')
    
    import keras
    print(f'  ✓ Keras: {keras.__version__}')
    
    from tensorflow.keras.models import load_model
    from tensorflow.keras.metrics import R2Score
    print('  ✓ TensorFlow Keras modules')
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    print('  ✓ Scikit-learn modules')
    
    from collections import deque
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    print('  ✓ Matplotlib and SciPy')
    
    import progressbar
    print('  ✓ Progressbar')
    
    import glob
    print('  ✓ Glob')
    
    # Check custom modules (optional)
    try:
        from utils import *
        print('  ✓ utils.py loaded')
    except ImportError:
        print('  ⚠ utils.py not found (optional)')
    
    try:
        from model import CNN_GRU, baseModel, CNN_LSTM
        print('  ✓ model.py loaded')
    except ImportError:
        print('  ⚠ model.py not found (optional)')
    
    try:
        from graph_navagation import NavigationGraphPredictor
        print('  ✓ graph_navagation.py loaded')
    except ImportError:
        print('  ⚠ graph_navagation.py not found (optional)')
    
    print('\n✅ All required modules imported successfully!')
    print(f'   Keras version: {keras.__version__} (Target: 2.13.1)')
    
    # Verify Keras version matches target
    if keras.__version__ != '2.13.1':
        print(f'⚠ WARNING: Keras version is {keras.__version__}, expected 2.13.1')
        sys.exit(1)
    else:
        print('✅ Keras version matches target (2.13.1)')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed successfully!${NC}"
else
    echo -e "${RED}⚠ Some tests failed. Please check the output above.${NC}"
    echo -e "${YELLOW}Try reinstalling with:${NC}"
    echo -e "  source $VENV_DIR/bin/activate"
    echo -e "  pip uninstall keras tensorflow -y"
    echo -e "  pip install tensorflow==2.13.0 keras==2.13.1"
fi

# Create a sample script template
cat > sample_script.py << EOF
#!/usr/bin/env python
# Sample script with Keras 2.13.1 and your configurations

import os
import sys
import numpy as np
import pandas as pd
import progressbar

# Your environment configurations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_NUM_INTEROP_THREADS'] = '2' 
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import R2Score

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Other imports
from collections import deque
import matplotlib.pyplot as plt
import scipy.stats as stats
import glob

# Your custom modules (uncomment when available)
# from utils import *
# from model import CNN_GRU, baseModel, CNN_LSTM
# from graph_navagation import NavigationGraphPredictor

# Print version information
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Your code here
print("Environment ready for Keras 2.13.1!")
EOF

chmod +x sample_script.py
echo -e "${GREEN}✓ Sample script created: sample_script.py${NC}"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}================================${NC}"
