@echo off
REM FloatChat Environment Setup Script for Windows
REM Sets up complete development and production environment

echo 🌊 FloatChat Environment Setup Starting...

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python %python_version% detected

REM Create virtual environment
echo 📦 Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 📈 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing Python packages...
pip install -r requirements.txt

REM Create data directories
echo 📁 Creating data directories...
if not exist "data" mkdir data
if not exist "data\netcdf" mkdir data\netcdf
if not exist "data\chromadb" mkdir data\chromadb
if not exist "logs" mkdir logs
echo ✅ Directories created

REM Check MongoDB connection (optional)
echo 🔍 Checking system requirements...
where mongod >nul 2>&1
if errorlevel 1 (
    echo ⚠️  MongoDB CLI not found - ensure MongoDB is installed and accessible
) else (
    echo ✅ MongoDB CLI detected
)

REM Set environment variables
echo 🔧 Setting up environment...
if not exist ".env" (
    echo # FloatChat Configuration > .env
    echo GROQ_API_KEY=your_groq_api_key_here >> .env
    echo MONGODB_URI=mongodb://localhost:27017 >> .env
    echo MONGODB_DATABASE=argo_data >> .env
    echo MONGODB_COLLECTION=float_profiles >> .env
    echo CHROMADB_PATH=./data/chromadb >> .env
    echo LOG_LEVEL=INFO >> .env
    echo ✅ .env file created - please update with your API keys
) else (
    echo ✅ .env file already exists
)

REM Validate configuration
echo 🧪 Validating configuration...
python -c "try: from config import config; print('✅ Configuration loaded successfully'); print(f'   - Database: {config.MONGODB_DATABASE}'); print(f'   - Collection: {config.MONGODB_COLLECTION}'); print(f'   - ChromaDB Path: {config.CHROMADB_PATH}'); print(f'   - Model: {config.GROQ_MODEL}'); except Exception as e: print(f'❌ Configuration error: {e}'); exit(1)"

echo.
echo 🎉 FloatChat setup completed successfully!
echo.
echo Next steps:
echo 1. Update .env with your Groq API key
echo 2. Ensure MongoDB is running
echo 3. Activate virtual environment: venv\Scripts\activate
echo 4. Run the application: streamlit run app.py
echo.
echo Test queries to try:
echo - 'Show me temperature profiles near Mumbai'
echo - 'Find floats in Arabian Sea from last 3 months'
echo - 'Compare salinity between regions'
echo.
pause