# 🌊 FloatChat - AI-Powered ARGO Data Assistant

**An advanced conversational system for ARGO oceanographic data discovery, analysis, and visualization**

## 🎯 Overview

FloatChat is a production-ready, AI-powered conversational interface that democratizes access to complex oceanographic data from the ARGO float network. Built with cutting-edge technologies including Groq AI, MongoDB, and ChromaDB, it enables natural language queries for ocean data analysis and visualization.

### Key Features

- **🤖 Natural Language Processing**: Ask questions in plain English about oceanographic data
- **🌍 Real-time Data Integration**: Direct NetCDF file processing from ARGO FTP servers
- **📊 Advanced Visualizations**: Interactive depth profiles, T-S diagrams, and geospatial maps
- **🔍 RAG-Enhanced AI**: Retrieval-Augmented Generation for contextual oceanographic insights
- **🗄️ Unified Database**: MongoDB for main data + ChromaDB for vector search
- **📤 Multiple Export Formats**: ASCII CSV and NetCDF export capabilities
- **🌊 BGC Parameter Support**: Biogeochemical data including oxygen, chlorophyll, nitrate

## 🏗️ Architecture

```
FloatChat System Architecture
├── Frontend (Streamlit)
│   ├── Interactive chat interface
│   ├── Real-time visualizations
│   └── Export controls
├── AI Engine (Groq + RAG)
│   ├── Natural language understanding
│   ├── MongoDB query generation
│   └── Context-aware responses
├── Data Layer
│   ├── MongoDB (geospatial + temporal data)
│   ├── ChromaDB (vector search)
│   └── NetCDF processor
└── Real-time Integration
    ├── ARGO FTP data ingestion
    ├── Quality flag processing
    └── Multi-format export
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MongoDB (local or cloud)
- Groq API key

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository>
cd floatchat
chmod +x setup.sh
./setup.sh  # Linux/Mac
# OR
setup.bat   # Windows
```

2. **Configure API keys:**
```bash
cp .env.example .env
# Edit .env with your Groq API key
```

3. **Run the application:**
```bash
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

streamlit run app.py
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# AI Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
MAX_TOKENS=2000
TEMPERATURE=0.1

# Database Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=argo_data
MONGODB_COLLECTION=float_profiles
CHROMADB_PATH=./data/chromadb

# Application Settings
LOG_LEVEL=INFO
AUTO_GENERATE_SAMPLES=true
```

### Geographic Regions

The system focuses on Indian Ocean regions:

- **Arabian Sea**: 50-80°E, 8-25°N
- **Bay of Bengal**: 80-100°E, 5-22°N
- **Indian Ocean**: 40-120°E, -20-30°N

## 🎯 Usage Examples

### Natural Language Queries

```
🌊 Basic Queries:
• "Show me temperature profiles near Mumbai"
• "Find floats in Arabian Sea from last 3 months"
• "What's the average salinity in Bay of Bengal?"

🧪 BGC Parameter Queries:
• "Show oxygen levels near the equator"
• "Compare chlorophyll between Arabian Sea and Bay of Bengal"
• "Find nitrate data in Bay of Bengal"

📊 Statistical Queries:
• "Average temperature at 100m depth"
• "Count active floats by region"
• "Temperature trends over time"

📤 Export Queries:
• "Export Mumbai temperature data to ASCII"
• "Save BGC data as NetCDF"
```

### MongoDB Query Generation

The AI engine converts natural language to MongoDB aggregation pipelines:

**Input**: "Show floats near Mumbai"
**Output**:
```javascript
[
  {
    "$match": {
      "location": {
        "$near": {
          "$geometry": {
            "type": "Point",
            "coordinates": [72.8777, 19.0760]
          },
          "$maxDistance": 100000
        }
      }
    }
  }
]
```

## 📊 Data Schema

### MongoDB Document Structure

```javascript
{
  "_id": "float_2901623_20230315",
  "float_id": "2901623",
  "cycle_number": 145,
  "location": {
    "type": "Point",
    "coordinates": [longitude, latitude]
  },
  "timestamp": ISODate("2023-03-15T06:30:00Z"),
  "region": "Arabian Sea",
  "platform_type": "APEX",
  "data_mode": "R",
  "measurements": [
    {
      "depth": 10.5,
      "pressure": 11.2,
      "temperature": 28.7,
      "salinity": 36.1,
      "oxygen": 245.3,        // BGC floats
      "chlorophyll": 0.12,    // BGC floats
      "nitrate": 1.8          // BGC floats
    }
  ],
  "quality_flags": {
    "temperature": "good",
    "salinity": "good"
  },
  "metadata": {
    "institution": "INCOIS",
    "project": "ARGO_INDIA",
    "netcdf_file": "original_file.nc"
  }
}
```

## 🔬 Technical Implementation

### AI Engine with RAG Pipeline

1. **Query Understanding**: Intent analysis and entity extraction
2. **Context Retrieval**: ChromaDB vector search for oceanographic knowledge
3. **Pipeline Generation**: Groq LLM converts natural language to MongoDB queries
4. **Response Enhancement**: Context-aware result interpretation

### Real-time Data Processing

```python
# NetCDF Processing Pipeline
processor = ARGONetCDFProcessor()
files = await processor.download_argo_index()
documents = await processor.process_multiple_files(files)
success = db_handler.insert_argo_data(documents)
```

### Visualization Engine

- **Depth Profiles**: Interactive Plotly charts with parameter selection
- **Geospatial Maps**: Folium maps with float trajectories
- **T-S Diagrams**: Temperature-Salinity relationship plots
- **BGC Multi-Parameter**: Oxygen, chlorophyll, nitrate analysis

## 📁 Project Structure

```
floatchat/
├── core/                   # Core system modules
│   ├── ai_engine.py       # Groq AI + RAG pipeline
│   ├── database.py        # MongoDB + ChromaDB handler
│   ├── netcdf_processor.py # Real-time ARGO data processor
│   └── visualizations.py  # Interactive visualization engine
├── config.py              # Centralized configuration
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── setup.py               # Package installation
├── setup.sh/.bat         # Environment setup scripts
└── .env.example          # Configuration template
```

---

**FloatChat** - Meeting the problem statement requirements through AI-powered oceanographic data democratization.