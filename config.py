"""
FloatChat Configuration Management
Centralized configuration for the entire application
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""

    # Project paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    NETCDF_DIR = DATA_DIR / "netcdf"
    PROCESSED_DIR = DATA_DIR / "processed"
    CACHE_DIR = DATA_DIR / "cache"

    # Database configuration
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'floatchat_production')
    MONGODB_COLLECTION = 'argo_floats'

    # ChromaDB configuration
    CHROMADB_PATH = os.getenv('CHROMADB_PATH', './chromadb_production')
    CHROMADB_COLLECTION = 'argo_context'

    # AI/LLM configuration
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.1-80b-instant')
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '4000'))
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.1'))

    # ARGO data sources
    ARGO_FTP_HOST = 'ftp.ifremer.fr'
    ARGO_FTP_PATH = '/ifremer/argo'
    INCOIS_BASE_URL = 'https://incois.gov.in/OON'

    # Data processing
    MAX_CONCURRENT_DOWNLOADS = int(os.getenv('MAX_CONCURRENT_DOWNLOADS', '5'))
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    UPDATE_INTERVAL_HOURS = int(os.getenv('UPDATE_INTERVAL_HOURS', '6'))

    # Geographic regions of interest
    REGIONS = {
        'Arabian Sea': {
            'bounds': {'lat': (8.0, 25.0), 'lon': (50.0, 80.0)},
            'center': (17.0, 65.0)
        },
        'Bay of Bengal': {
            'bounds': {'lat': (5.0, 22.0), 'lon': (80.0, 100.0)},
            'center': (13.5, 90.0)
        },
        'Indian Ocean': {
            'bounds': {'lat': (-20.0, 30.0), 'lon': (40.0, 120.0)},
            'center': (5.0, 80.0)
        },
        'Equatorial Pacific': {
            'bounds': {'lat': (-10.0, 10.0), 'lon': (120.0, 280.0)},
            'center': (0.0, 200.0)
        }
    }

    # Major cities coordinates
    CITIES = {
        'mumbai': {'lat': 19.0760, 'lon': 72.8777, 'region': 'Arabian Sea'},
        'chennai': {'lat': 13.0827, 'lon': 80.2707, 'region': 'Bay of Bengal'},
        'kochi': {'lat': 9.9312, 'lon': 76.2673, 'region': 'Arabian Sea'},
        'kolkata': {'lat': 22.5726, 'lon': 88.3639, 'region': 'Bay of Bengal'},
        'goa': {'lat': 15.2993, 'lon': 73.8278, 'region': 'Arabian Sea'},
        'visakhapatnam': {'lat': 17.6868, 'lon': 83.2185, 'region': 'Bay of Bengal'},
        'delhi': {'lat': 28.6139, 'lon': 77.2090, 'region': 'Land'},
        'bangalore': {'lat': 12.9716, 'lon': 77.5946, 'region': 'Land'}
    }

    # Oceanographic parameters
    PARAMETERS = {
        'CTD': ['temperature', 'salinity', 'pressure', 'depth'],
        'BGC': ['oxygen', 'chlorophyll', 'nitrate', 'ph', 'fluorescence', 'turbidity'],
        'DERIVED': ['density', 'sound_velocity', 'potential_temperature']
    }

    # Quality flags
    QUALITY_FLAGS = {
        1: 'good',
        2: 'probably_good',
        3: 'probably_bad',
        4: 'bad',
        5: 'changed',
        8: 'estimated',
        9: 'missing'
    }

    # Streamlit configuration
    STREAMLIT_CONFIG = {
        'page_title': '🌊 FloatChat - ARGO Data Assistant',
        'page_icon': '🌊',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }

    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Cache settings
    CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', '3600'))
    MAX_CACHE_SIZE_MB = int(os.getenv('MAX_CACHE_SIZE_MB', '500'))

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.DATA_DIR, cls.NETCDF_DIR, cls.PROCESSED_DIR, cls.CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_config(cls):
        """Validate configuration"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required")

        cls.create_directories()
        return True

# Global config instance
config = Config()

# Validate on import
config.validate_config()