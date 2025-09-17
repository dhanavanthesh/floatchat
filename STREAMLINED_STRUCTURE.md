# 🌊 FloatChat - Streamlined File Structure

## 📁 **Minimal Production Files**

```
floatchat/                          # Root directory
├── 🎨 FRONTEND + BACKEND
│   ├── app.py                      # Main Streamlit application (UI + orchestration)
│   └── config.py                   # Configuration management
│
├── 🔧 BACKEND MODULES
│   └── core/
│       ├── ai_engine.py           # Groq LLM + RAG pipeline
│       ├── database.py            # MongoDB + ChromaDB handler
│       ├── netcdf_processor.py    # Real-time ARGO data processing
│       └── visualizations.py      # Interactive charts and maps
│
├── ⚙️ SETUP & CONFIG
│   ├── setup.py                   # Package installation
│   ├── setup.sh                   # Linux/Mac setup script
│   ├── setup.bat                  # Windows setup script
│   ├── requirements.txt           # Python dependencies
│   ├── .env.example              # Configuration template
│   └── .env                      # Your API keys (private)
│
├── 📊 DATA & STORAGE
│   ├── data/                     # Data directory (auto-created)
│   └── chromadb_storage/         # ChromaDB files (auto-created)
│
└── 📚 DOCUMENTATION
    ├── PROBLEM_STATEMENT_COMPLIANCE.md
    └── SYSTEM_OVERVIEW.md
```

## 🚀 **How It Works - Single App Architecture**

### **1. Frontend UI (Streamlit Interface)**
**File**: `app.py`
- Chat interface for natural language queries
- Interactive visualizations display
- Export controls and data tables
- System status and controls

### **2. Backend Logic (Embedded Modules)**
**Files**: `core/*.py`
- `ai_engine.py` - Natural language processing
- `database.py` - Data storage and retrieval
- `netcdf_processor.py` - Real-time data processing
- `visualizations.py` - Chart generation

### **3. Configuration**
**File**: `config.py`
- Database connections
- API keys and models
- Geographic regions
- System parameters

## ⚡ **Run Command (Single Application)**

```bash
# 1. Setup (one time)
./setup.sh        # Linux/Mac
setup.bat         # Windows

# 2. Run (every time)
streamlit run app.py
```

## 🔄 **Data Flow in Single App**

```
User Query (Frontend)
    ↓
app.py captures input
    ↓
ai_engine.py processes → MongoDB pipeline
    ↓
database.py executes → retrieves data
    ↓
visualizations.py generates → creates charts
    ↓
app.py displays (Frontend) → shows results
```

## 📦 **Essential Files Only (No Redundancy)**

| **Component** | **File** | **Purpose** |
|---------------|----------|-------------|
| **Main App** | `app.py` | Frontend UI + Backend orchestration |
| **AI Engine** | `core/ai_engine.py` | Natural language → MongoDB queries |
| **Database** | `core/database.py` | MongoDB + ChromaDB operations |
| **Data Processing** | `core/netcdf_processor.py` | NetCDF processing + export |
| **Visualizations** | `core/visualizations.py` | Interactive charts and maps |
| **Configuration** | `config.py` | System settings |

## ✅ **Complete Functionality in Minimal Files**

- ✅ Real-time ARGO NetCDF processing
- ✅ Natural language query processing
- ✅ MongoDB + ChromaDB database
- ✅ Interactive visualizations
- ✅ ASCII + NetCDF export
- ✅ RAG-enhanced AI responses
- ✅ Geospatial and temporal queries
- ✅ BGC parameter support

**Total Core Files: 6 Python files**
**Total Setup Files: 5 configuration files**
**No separate backend/frontend servers needed!**

---

This is the **minimal, production-ready FloatChat system** that meets all problem statement requirements in the fewest possible files.