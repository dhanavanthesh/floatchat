#!/bin/bash

# FloatChat Environment Setup Script
# Sets up complete development and production environment

set -e  # Exit on any error

echo "🌊 FloatChat Environment Setup Starting..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/netcdf
mkdir -p data/chromadb
mkdir -p logs
echo "✅ Directories created"

# Check MongoDB connection (optional)
echo "🔍 Checking system requirements..."

# Check if MongoDB is available
if command -v mongod &> /dev/null; then
    echo "✅ MongoDB CLI detected"
else
    echo "⚠️  MongoDB CLI not found - ensure MongoDB is installed and accessible"
fi

# Set environment variables
echo "🔧 Setting up environment..."
if [ ! -f ".env" ]; then
    cat > .env << EOL
# FloatChat Configuration
GROQ_API_KEY=your_groq_api_key_here
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=argo_data
MONGODB_COLLECTION=float_profiles
CHROMADB_PATH=./data/chromadb
LOG_LEVEL=INFO
EOL
    echo "✅ .env file created - please update with your API keys"
else
    echo "✅ .env file already exists"
fi

# Validate configuration
echo "🧪 Validating configuration..."
python3 -c "
try:
    from config import config
    print('✅ Configuration loaded successfully')
    print(f'   - Database: {config.MONGODB_DATABASE}')
    print(f'   - Collection: {config.MONGODB_COLLECTION}')
    print(f'   - ChromaDB Path: {config.CHROMADB_PATH}')
    print(f'   - Model: {config.GROQ_MODEL}')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    exit(1)
"

echo ""
echo "🎉 FloatChat setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Update .env with your Groq API key"
echo "2. Ensure MongoDB is running"
echo "3. Activate virtual environment: source venv/bin/activate"
echo "4. Run the application: streamlit run app.py"
echo ""
echo "Test queries to try:"
echo "- 'Show me temperature profiles near Mumbai'"
echo "- 'Find floats in Arabian Sea from last 3 months'"
echo "- 'Compare salinity between regions'"
echo ""