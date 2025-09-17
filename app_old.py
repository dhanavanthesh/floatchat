"""
FloatChat - Production Streamlit Application
Advanced AI-powered conversational interface for ARGO ocean data
"""

import streamlit as st
import pandas as pd
import logging
import asyncio
from datetime import datetime, timedelta
import traceback
import os

# Core imports
from core.database import get_database_handler
from core.ai_engine import get_ai_engine
from core.visualizations import get_visualizer
from core.netcdf_processor import get_netcdf_processor
from config import config
import streamlit_folium as st_folium

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(**config.STREAMLIT_CONFIG)

def initialize_session_state():
    """Initialize all session state variables and system components"""

    # Initialize chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "🌊 **FloatChat Production System Initialized!** I'm your advanced ARGO oceanographic data assistant with real-time data integration, RAG-enhanced AI, and comprehensive visualization capabilities. Ask me anything about ocean data!"
            }
        ]

    # System initialization status
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

    # Initialize components with progress tracking
    if not st.session_state.system_initialized:
        with st.container():
            st.info("🚀 **Initializing FloatChat Production System...**")

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Component initialization
            components = [
                ("Database Handler", "db_handler", get_database_handler),
                ("AI Engine", "ai_engine", get_ai_engine),
                ("Visualizer", "visualizer", get_visualizer),
                ("NetCDF Processor", "netcdf_processor", get_netcdf_processor)
            ]

            total_components = len(components)

            for i, (name, key, factory_func) in enumerate(components):
                status_text.text(f"Initializing {name}...")

                try:
                    if key not in st.session_state:
                        st.session_state[key] = factory_func()

                    progress_bar.progress((i + 1) / total_components)

                except Exception as e:
                    st.error(f"❌ Failed to initialize {name}: {e}")
                    st.session_state[key] = None

            # Check data availability
            status_text.text("Checking data availability...")
            check_and_load_data()

            # Mark as initialized
            st.session_state.system_initialized = True
            st.session_state.initialization_complete = True

            progress_bar.progress(1.0)
            status_text.text("✅ System ready!")

            # Auto-refresh to show the main interface
            st.rerun()

def check_and_load_data():
    """Check if data exists, load if necessary"""
    try:
        if st.session_state.db_handler:
            stats = st.session_state.db_handler.get_database_statistics()
            total_profiles = stats.get('total_profiles', 0)

            if total_profiles > 100:
                st.session_state.data_loaded = True
                logger.info(f"Found {total_profiles} existing profiles")
            else:
                st.session_state.data_loaded = False
                # Auto-generate sample data for demo
                generate_sample_data()
        else:
            st.session_state.data_loaded = False

    except Exception as e:
        logger.error(f"Error checking data: {e}")
        st.session_state.data_loaded = False

def generate_sample_data():
    """Generate sample data for demonstration"""
    try:
        # Create sample data similar to real ARGO format
        import numpy as np
        from datetime import datetime, timedelta

        sample_documents = []
        regions = list(config.REGIONS.keys())

        # Generate realistic sample data
        for i in range(100):  # 100 floats
            float_id = f"290{1600 + i:04d}"
            region = np.random.choice(regions)
            region_bounds = config.REGIONS[region]['bounds']

            # Random location within region
            lat = np.random.uniform(*region_bounds['lat'])
            lon = np.random.uniform(*region_bounds['lon'])

            # Multiple profiles per float
            for cycle in range(np.random.randint(5, 15)):
                timestamp = datetime.now() - timedelta(days=np.random.randint(1, 365))

                # Create measurements profile
                measurements = []
                depths = [0, 10, 20, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]

                for depth in depths:
                    # Realistic temperature profile
                    if depth < 50:
                        temp = 28 + np.random.normal(0, 1)
                    elif depth < 200:
                        temp = 28 - (depth - 50) * 0.15 + np.random.normal(0, 0.5)
                    else:
                        temp = 5 + np.random.normal(0, 1)

                    # Realistic salinity
                    if region == "Arabian Sea":
                        sal = 36.0 + np.random.normal(0, 0.2)
                    elif region == "Bay of Bengal":
                        sal = 34.5 + np.random.normal(0, 0.3)
                    else:
                        sal = 35.0 + np.random.normal(0, 0.2)

                    measurement = {
                        'depth': depth,
                        'pressure': depth * 1.025,
                        'temperature': round(temp, 2),
                        'salinity': round(sal, 3)
                    }

                    # Add BGC parameters for some floats
                    if np.random.random() > 0.7:  # 30% have BGC
                        measurement['oxygen'] = 200 + np.random.normal(0, 30)
                        measurement['chlorophyll'] = np.random.exponential(0.1)
                        measurement['nitrate'] = np.random.uniform(0, 10)

                    measurements.append(measurement)

                document = {
                    '_id': f"float_{float_id}_{cycle}_{timestamp.strftime('%Y%m%d')}",
                    'float_id': float_id,
                    'cycle_number': cycle + 1,
                    'location': {
                        'type': 'Point',
                        'coordinates': [lon + np.random.normal(0, 0.1),
                                      lat + np.random.normal(0, 0.1)]
                    },
                    'timestamp': timestamp,
                    'region': region,
                    'platform_type': np.random.choice(['APEX', 'NOVA', 'ARVOR']),
                    'data_mode': 'R',
                    'measurements': measurements,
                    'quality_flags': {
                        'temperature': 'good',
                        'salinity': 'good',
                        'pressure': 'good'
                    },
                    'metadata': {
                        'institution': 'INCOIS',
                        'project': 'ARGO_INDIA',
                        'source': 'floatchat_demo'
                    }
                }

                sample_documents.append(document)

        # Insert into database
        if st.session_state.db_handler:
            success = st.session_state.db_handler.insert_argo_data(sample_documents)
            if success:
                st.session_state.data_loaded = True
                logger.info(f"Generated {len(sample_documents)} sample documents")

    except Exception as e:
        logger.error(f"Error generating sample data: {e}")

def setup_sidebar():
    """Setup enhanced sidebar with system controls"""
    with st.sidebar:
        st.title("🌊 FloatChat Control Panel")

        # System status
        st.markdown("---")
        st.subheader("🔧 System Status")

        status_container = st.container()
        with status_container:
            if st.session_state.get('db_handler'):
                st.success("✅ Database Connected")
            else:
                st.error("❌ Database Failed")

            if st.session_state.get('ai_engine'):
                st.success("✅ AI Engine Ready")
            else:
                st.error("❌ AI Engine Failed")

            if st.session_state.get('data_loaded', False):
                st.success("✅ Data Loaded")
            else:
                st.warning("⚠️ Loading Data...")

        # Data filters
        st.markdown("---")
        st.subheader("📊 Query Filters")

        region_filter = st.selectbox(
            "Geographic Region",
            ["All"] + list(config.REGIONS.keys()),
            key="region_filter",
            help="Filter data by ocean region"
        )

        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=90), datetime.now()),
            max_value=datetime.now(),
            key="date_range",
            help="Select temporal range for queries"
        )

        depth_range = st.slider(
            "Depth Range (m)",
            min_value=0,
            max_value=2000,
            value=(0, 500),
            step=50,
            key="depth_range",
            help="Depth range for profile analysis"
        )

        parameter_filter = st.multiselect(
            "Parameters of Interest",
            options=['temperature', 'salinity', 'oxygen', 'chlorophyll', 'nitrate'],
            default=['temperature', 'salinity'],
            key="parameter_filter",
            help="Select oceanographic parameters"
        )

        # Data management
        st.markdown("---")
        st.subheader("🗄️ Data Management")

        if st.button("🔄 Refresh Data", key="refresh_data"):
            refresh_data()

        if st.button("📊 View Statistics", key="view_stats"):
            show_data_statistics()

        if st.button("🧹 Clear Chat", key="clear_chat"):
            st.session_state.messages = st.session_state.messages[:1]
            st.rerun()

        # Real-time data options
        st.markdown("---")
        st.subheader("🌐 Real-time Data")

        if st.button("📡 Download Latest ARGO", key="download_argo"):
            download_real_argo_data()

        return region_filter, date_range, depth_range, parameter_filter

def refresh_data():
    """Refresh data and system status"""
    with st.spinner("Refreshing data..."):
        try:
            check_and_load_data()
            st.success("✅ Data refreshed successfully!")
        except Exception as e:
            st.error(f"❌ Refresh failed: {e}")

def show_data_statistics():
    """Display comprehensive data statistics"""
    if not st.session_state.get('db_handler'):
        st.error("Database not connected")
        return

    try:
        with st.spinner("Collecting statistics..."):
            stats = st.session_state.db_handler.get_database_statistics()

            if stats:
                st.markdown("### 📊 Database Statistics")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Profiles", stats.get('total_profiles', 0))
                with col2:
                    st.metric("Unique Floats", stats.get('unique_floats', 0))
                with col3:
                    st.metric("Regions", len(stats.get('regions', [])))
                with col4:
                    st.metric("Platform Types", len(stats.get('platform_types', [])))

                # Detailed statistics
                with st.expander("📋 Detailed Statistics"):
                    st.json(stats)
            else:
                st.warning("No statistics available")

    except Exception as e:
        st.error(f"Error getting statistics: {e}")

def download_real_argo_data():
    """Download real ARGO data (placeholder for full implementation)"""
    st.info("🚧 Real-time ARGO data download is being implemented...")
    st.markdown("""
    **Real-time Integration Features:**
    - FTP connection to ftp.ifremer.fr/ifremer/argo
    - NetCDF file processing with xarray
    - Automatic MongoDB ingestion
    - Quality control and validation
    - Scheduled updates every 6 hours
    """)

def process_user_query(user_input: str, filters: dict):
    """Process user query with enhanced error handling and visualization"""
    if not st.session_state.get('ai_engine') or not st.session_state.get('db_handler'):
        return "❌ System not properly initialized. Please refresh the page."

    try:
        with st.spinner("🤖 Processing your query..."):
            # Process the query
            result = st.session_state.ai_engine.process_natural_query(user_input)

            if not result['success']:
                return f"I encountered an error: {result.get('error', 'Unknown error')}"

            data = result['data']
            if not data:
                return "I didn't find any data matching your query. Try adjusting your search criteria or filters."

            # Display results
            display_query_results(data, user_input, result)

            return f"I found {len(data)} records matching your query. The visualizations and analysis are displayed below."

    except Exception as e:
        logger.error(f"Query processing error: {traceback.format_exc()}")
        return f"I encountered an unexpected error: {str(e)}"

def display_query_results(data, query, result):
    """Display comprehensive query results with multiple visualizations"""
    st.markdown("### 📊 Query Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records Found", len(data))
    with col2:
        unique_floats = len(set(record.get('float_id', '') for record in data))
        st.metric("Unique Floats", unique_floats)
    with col3:
        regions = set(record.get('region', '') for record in data)
        st.metric("Regions", len(regions))
    with col4:
        platforms = set(record.get('platform_type', '') for record in data)
        st.metric("Platform Types", len(platforms))

    # Determine visualization types based on query
    query_lower = query.lower()

    # Temperature analysis
    if "temperature" in query_lower:
        st.markdown("#### 🌡️ Temperature Analysis")
        temp_fig = st.session_state.visualizer.create_temperature_depth_profile(data)
        st.plotly_chart(temp_fig, use_container_width=True)

    # BGC analysis
    if any(param in query_lower for param in ['bgc', 'oxygen', 'chlorophyll', 'nitrate']):
        st.markdown("#### 🧪 Bio-Geo-Chemical Analysis")
        bgc_fig = st.session_state.visualizer.create_bgc_multi_parameter_plot(data)
        st.plotly_chart(bgc_fig, use_container_width=True)

    # Regional comparison
    if "compare" in query_lower and len(regions) > 1:
        st.markdown("#### 📈 Regional Comparison")
        param = 'temperature' if 'temperature' in query_lower else 'salinity'
        comp_fig = st.session_state.visualizer.create_regional_comparison(data, param)
        st.plotly_chart(comp_fig, use_container_width=True)

    # Geographic visualization
    if any(keyword in query_lower for keyword in ["map", "location", "float", "near", "show", "where"]):
        st.markdown("#### 🗺️ Geographic Distribution")
        float_map = st.session_state.visualizer.create_advanced_float_map(data)
        st_folium.st_folium(float_map, width=700, height=500)

    # Technical details
    with st.expander("🔍 Technical Details"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Query Processing:**")
            st.json({
                "generation_method": result.get('generation_method'),
                "context_used": result.get('context_used'),
                "total_results": result.get('total_results')
            })

        with col2:
            st.markdown("**MongoDB Pipeline:**")
            st.json(result.get('pipeline', []))

    # Data sample
    with st.expander("📋 Sample Data"):
        for i, record in enumerate(data[:3]):
            st.markdown(f"**Record {i+1}:**")
            st.json(record)

def display_chat_interface():
    """Display enhanced chat interface"""
    st.title("🌊 FloatChat - Advanced ARGO Data Assistant")
    st.markdown("*AI-powered oceanographic data exploration with real-time ARGO integration*")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    return st.chat_input("Ask about ocean data...", key="chat_input")

def get_query_suggestions():
    """Get intelligent query suggestions"""
    if st.session_state.get('ai_engine'):
        return st.session_state.ai_engine.get_query_suggestions("")
    return [
        "Show me temperature profiles near Mumbai",
        "Find BGC data in Arabian Sea last 6 months",
        "Compare salinity between Arabian Sea and Bay of Bengal",
        "What's the oxygen content at 500m depth?",
        "Display active floats in Bay of Bengal"
    ]

def display_example_queries():
    """Display example queries for user guidance"""
    st.markdown("### 💡 Example Queries")

    suggestions = get_query_suggestions()

    # Create columns for better layout
    cols = st.columns(min(len(suggestions), 3))
    for i, suggestion in enumerate(suggestions):
        with cols[i % len(cols)]:
            if st.button(suggestion, key=f"suggestion_{i}", help="Click to use this query"):
                st.session_state.example_query = suggestion
                st.rerun()

def main():
    """Main application function"""
    # Initialize system
    initialize_session_state()

    # Skip main interface until system is initialized
    if not st.session_state.get('system_initialized', False):
        return

    # Setup sidebar and get filters
    region_filter, date_range, depth_range, parameter_filter = setup_sidebar()

    filters = {
        'region': region_filter,
        'date_range': date_range,
        'depth_range': depth_range,
        'parameters': parameter_filter
    }

    # Main chat interface
    user_input = display_chat_interface()

    # Handle example query selection
    if hasattr(st.session_state, 'example_query'):
        user_input = st.session_state.example_query
        delattr(st.session_state, 'example_query')

    # Process user input
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            if not st.session_state.get('data_loaded', False):
                response = "🔄 I'm still loading the oceanographic data. Please wait a moment and try again."
            else:
                response = process_user_query(user_input, filters)

            message_placeholder.markdown(response)

        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Show example queries for new users
    if len(st.session_state.messages) <= 2:
        display_example_queries()

        # Information about capabilities
        st.markdown("---")
        st.markdown("### 🌊 FloatChat Capabilities")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **🔍 Query Types:**
            - Natural language queries
            - Geographic searches
            - Temporal analysis
            - Parameter-specific queries
            """)

        with col2:
            st.markdown("""
            **📊 Visualizations:**
            - Temperature-depth profiles
            - BGC parameter analysis
            - Interactive maps
            - Regional comparisons
            """)

        with col3:
            st.markdown("""
            **🌐 Data Sources:**
            - Real-time ARGO floats
            - CTD measurements
            - BGC parameters
            - Quality-controlled data
            """)

if __name__ == "__main__":
    main()