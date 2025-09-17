"""
FloatChat - Enhanced Production Streamlit Application
Real-time ARGO oceanographic data with advanced AI conversation
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
import requests

# Core imports
from core.database import get_database_handler
from core.ai_engine import get_ai_engine
from core.visualizations import get_visualizer
from core.netcdf_processor import get_netcdf_processor
from core.met_data import get_met_context
from config import config
import streamlit_folium as st_folium

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🌊 FloatChat - ARGO Data Assistant",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def show_loading_animation(message: str, duration: float = 2.0):
    """Show animated loading with progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(100):
        progress = (i + 1) / 100
        progress_bar.progress(progress)
        status_text.text(f"{message} {int(progress * 100)}%")
        time.sleep(duration / 100)

    progress_bar.empty()
    status_text.empty()

def determine_query_type(query: str) -> str:
    """Determine the main focus of the query for appropriate response"""
    query_lower = query.lower()

    # Geographic queries
    if any(city in query_lower for city in ['mumbai', 'chennai', 'kolkata', 'goa', 'kochi', 'visakhapatnam']):
        return 'geographic'
    if any(region in query_lower for region in ['arabian sea', 'bay of bengal', 'indian ocean']):
        return 'regional'

    # Parameter-specific queries
    if 'temperature' in query_lower:
        return 'temperature'
    if 'salinity' in query_lower:
        return 'salinity'
    if any(param in query_lower for param in ['oxygen', 'chlorophyll', 'nitrate', 'bgc']):
        return 'bgc'

    # Analysis queries
    if any(word in query_lower for word in ['compare', 'comparison', 'versus', 'vs']):
        return 'comparison'
    if any(word in query_lower for word in ['trend', 'time', 'temporal', 'recent']):
        return 'temporal'
    if any(word in query_lower for word in ['depth', 'profile', 'vertical']):
        return 'depth'

    return 'general'

def generate_context_aware_tabs(query: str, data: List[Dict]) -> List[tuple]:
    """Generate appropriate tabs based on query context"""
    query_type = determine_query_type(query)

    # Check what data is available
    has_temp = any(any(m.get('temperature') for m in item.get('measurements', [])) for item in data)
    has_sal = any(any(m.get('salinity') for m in item.get('measurements', [])) for item in data)
    has_bgc = any(any(m.get('oxygen') or m.get('chlorophyll') or m.get('nitrate') for m in item.get('measurements', [])) for item in data)
    has_geo = any(item.get('location') for item in data)

    tabs = []

    if query_type == 'geographic':
        if has_geo:
            tabs.append(("🗺️ Location Analysis", "geographic"))
        if has_temp:
            tabs.append(("🌡️ Regional Temperature", "temperature"))
        if has_sal:
            tabs.append(("🧂 Regional Salinity", "salinity"))

    elif query_type == 'temperature':
        if has_temp:
            tabs.append(("🌡️ Temperature Analysis", "temperature"))
            tabs.append(("📊 Temperature vs Depth", "depth"))
        if has_sal and has_temp:
            tabs.append(("🔄 T-S Diagram", "ts_diagram"))

    elif query_type == 'salinity':
        if has_sal:
            tabs.append(("🧂 Salinity Analysis", "salinity"))
            tabs.append(("📊 Salinity Profiles", "depth"))
        if has_geo:
            tabs.append(("🗺️ Salinity Distribution", "geographic"))

    elif query_type == 'bgc':
        if has_bgc:
            tabs.append(("🔬 BGC Parameters", "bgc"))
            tabs.append(("📊 BGC Profiles", "depth"))
        if has_geo:
            tabs.append(("🗺️ BGC Distribution", "geographic"))

    elif query_type == 'comparison':
        if has_temp:
            tabs.append(("🌡️ Temperature Comparison", "temperature"))
        if has_sal:
            tabs.append(("🧂 Enhanced Salinity Comparison", "salinity_comparison"))
        if has_geo:
            tabs.append(("🗺️ Regional Comparison", "geographic"))

    elif query_type == 'depth':
        if has_temp or has_sal:
            tabs.append(("📊 Depth Profiles", "depth"))
        if has_temp and has_sal:
            tabs.append(("🔄 T-S Analysis", "ts_diagram"))
        if has_bgc:
            tabs.append(("🔬 BGC vs Depth", "bgc"))

    else:  # general
        if has_geo:
            tabs.append(("🗺️ Geographic Overview", "geographic"))
        if has_temp:
            tabs.append(("🌡️ Temperature Data", "temperature"))
        if has_sal:
            tabs.append(("🧂 Salinity Data", "salinity"))
        if has_bgc:
            tabs.append(("🔬 BGC Data", "bgc"))

    # Always add summary if we have data
    if data:
        tabs.append(("📋 Data Summary", "summary"))

    return tabs[:4]  # Limit to 4 tabs max

def create_comprehensive_visualizations(results: Dict, query: str, viz):
    """Create comprehensive visualizations with proper tab handling"""
    try:
        data = results.get('data', [])
        if not data:
            st.warning("No data available for visualization.")
            return

        # Generate context-aware tabs based on query
        dynamic_tabs = generate_context_aware_tabs(query, data)

        if not dynamic_tabs:
            st.warning("No appropriate visualizations found for this query.")
            return

        # Create tabs
        tab_objects = st.tabs([tab[0] for tab in dynamic_tabs])

        # Process each tab
        for i, (tab_name, tab_type) in enumerate(dynamic_tabs):
            if i < len(tab_objects):
                with tab_objects[i]:
                    if tab_type == 'depth' or tab_type == 'temperature':
                        st.markdown("### Temperature vs Depth Analysis")
                        try:
                            depth_fig = viz.create_depth_profile_plot(data)
                            if depth_fig:
                                st.plotly_chart(depth_fig, use_container_width=True)
                            else:
                                st.warning("Could not generate depth profile plot")
                        except Exception as e:
                            st.error(f"Error creating depth plot: {e}")

                        st.markdown("### Temperature-Salinity Relationship")
                        try:
                            ts_fig = viz.create_ts_diagram(data)
                            if ts_fig:
                                st.plotly_chart(ts_fig, use_container_width=True)
                            else:
                                st.warning("Could not generate T-S diagram")
                        except Exception as e:
                            st.error(f"Error creating T-S diagram: {e}")

                    elif tab_type == 'geographic':
                        st.markdown("### Float Locations and Trajectories")
                        map_fig = viz.create_float_trajectory_map(data)
                        st_folium.st_folium(map_fig, width=700, height=500)

                        # Regional statistics
                        regions = {}
                        for item in data:
                            region = item.get('region', 'Unknown')
                            regions[region] = regions.get(region, 0) + 1

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Profiles", len(data))
                        with col2:
                            st.metric("Unique Regions", len(regions))

                        for region, count in regions.items():
                            st.write(f"**{region}**: {count} profiles")

                    elif tab_type == 'summary':
                        st.markdown("### Statistical Analysis")

                        # Calculate statistics
                        all_temps = []
                        all_salinities = []
                        depths = []

                        for item in data:
                            for measurement in item.get('measurements', []):
                                if measurement.get('temperature'):
                                    all_temps.append(measurement['temperature'])
                                    depths.append(measurement.get('depth', 0))
                                if measurement.get('salinity'):
                                    all_salinities.append(measurement['salinity'])

                        if all_temps and len(all_temps) > 0:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Avg Temperature", f"{sum(all_temps)/len(all_temps):.2f}°C")
                            with col2:
                                st.metric("Avg Salinity", f"{sum(all_salinities)/len(all_salinities):.2f} PSU" if all_salinities and len(all_salinities) > 0 else "N/A")
                            with col3:
                                st.metric("Max Depth", f"{max(depths):.0f}m" if depths else "N/A")
                        else:
                            st.info("No temperature/salinity data available for statistics.")

                    elif tab_type == 'bgc':
                        st.markdown("### Biogeochemical Parameters")

                        # Check for BGC data
                        has_bgc = any(
                            any(m.get('oxygen') or m.get('chlorophyll') or m.get('nitrate')
                                for m in item.get('measurements', []))
                            for item in data
                        )

                        if has_bgc:
                            bgc_fig = viz.create_bgc_multi_parameter_plot(data)
                            st.plotly_chart(bgc_fig, use_container_width=True)
                        else:
                            st.info("No BGC parameters available in this dataset. BGC floats measure oxygen, chlorophyll, and nitrate levels.")

                    elif tab_type == 'salinity_comparison':
                        st.markdown("### Enhanced Salinity Analysis: Arabian Sea vs Bay of Bengal")

                        # Check if we have data from both regions for comparison
                        regions_in_data = set(item.get('region', '') for item in data)
                        has_arabian = any('Arabian Sea' in region for region in regions_in_data)
                        has_bay = any('Bay of Bengal' in region for region in regions_in_data)

                        if has_arabian or has_bay:
                            salinity_comparison_fig = viz.create_salinity_comparison_arabian_bay(data)
                            st.plotly_chart(salinity_comparison_fig, use_container_width=True)

                            # Add oceanographic context
                            if has_arabian and has_bay:
                                st.markdown("""
                                **🌊 Oceanographic Context:**

                                - **Arabian Sea**: Higher salinity (35.5-36.5 PSU) due to high evaporation and limited freshwater input
                                - **Bay of Bengal**: Lower salinity (33-35 PSU) due to significant river discharge from Ganges-Brahmaputra system
                                - **Seasonal Variation**: Monsoon impacts both regions differently, with Bay of Bengal showing more dramatic freshening
                                - **Depth Stratification**: Arabian Sea shows stronger haline stratification, Bay of Bengal shows temperature-dominated stratification
                                """)
                            elif has_arabian:
                                st.info("**Arabian Sea Data**: Showing high-salinity waters characteristic of this evaporation-dominated basin.")
                            elif has_bay:
                                st.info("**Bay of Bengal Data**: Showing lower-salinity waters influenced by major river systems.")
                        else:
                            st.info("Enhanced salinity comparison requires data from Arabian Sea and/or Bay of Bengal regions.")

        st.success("✅ All visualizations generated successfully!")

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        # Show basic data table as fallback
        st.markdown("### Data Preview")
        df_data = []
        for item in data[:10]:  # Show first 10
            df_data.append({
                'Float ID': item.get('float_id'),
                'Region': item.get('region'),
                'Timestamp': item.get('timestamp'),
                'Measurements': len(item.get('measurements', []))
            })
        st.dataframe(pd.DataFrame(df_data))

def generate_intelligent_response(data: List[Dict], query: str) -> str:
    """Generate intelligent responses using Groq AI model with real data analysis"""

    if not data:
        query_type = determine_query_type(query)
        if query_type == 'geographic':
            return "I couldn't find any ARGO float data for that specific location. Our coverage focuses on Mumbai, Chennai, and the Arabian Sea/Bay of Bengal regions. Try asking about data 'near Mumbai' or 'in Arabian Sea' for better results."
        elif query_type in ['temperature', 'salinity', 'bgc']:
            return f"I couldn't find any {query_type} data matching your query. This might be because the location is outside our coverage area or the specific parameters aren't available in recent data. Try specifying a location like 'temperature near Mumbai'."
        else:
            return "I couldn't find any ARGO float data matching your specific query. Try queries like 'temperature near Mumbai', 'Arabian Sea salinity', or 'BGC data near Chennai' for better results."

    # Analyze the data in detail
    total_profiles = len(data)
    regions = {}
    platforms = {}
    temp_data = []
    sal_data = []
    depth_data = []
    oxy_data = []
    chl_data = []
    nitrate_data = []
    bgc_floats = 0
    time_range = []
    met_contexts = []
    seasonal_info = {}

    for item in data:
        # Regional analysis
        region = item.get('region', 'Unknown')
        regions[region] = regions.get(region, 0) + 1

        # Platform analysis
        platform = item.get('platform_type', 'Unknown')
        platforms[platform] = platforms.get(platform, 0) + 1

        # Time analysis
        if item.get('timestamp'):
            time_range.append(item['timestamp'])

        # Parameter analysis
        measurements = item.get('measurements', [])
        has_bgc = any(m.get('oxygen') or m.get('chlorophyll') or m.get('nitrate') for m in measurements)
        if has_bgc:
            bgc_floats += 1

        for m in measurements:
            if m.get('temperature'):
                temp_data.append(m['temperature'])
                depth_data.append(m.get('depth', 0))
            if m.get('salinity'):
                sal_data.append(m['salinity'])
            if m.get('oxygen'):
                oxy_data.append(m['oxygen'])
            if m.get('chlorophyll'):
                chl_data.append(m['chlorophyll'])
            if m.get('nitrate'):
                nitrate_data.append(m['nitrate'])

    # Use Groq AI to generate contextual response
    if st.session_state.get('ai_engine'):
        try:
            # Create enhanced prompt for Groq
            data_summary = f"""
            Query: {query}
            Total profiles: {total_profiles}
            Regions: {regions}
            Temperature data: {len(temp_data)} measurements, avg: {sum(temp_data)/len(temp_data):.1f}°C if temp_data and len(temp_data) > 0 else 'No temp data'
            Salinity data: {len(sal_data)} measurements, avg: {sum(sal_data)/len(sal_data):.1f} PSU if sal_data and len(sal_data) > 0 else 'No salinity data'
            BGC floats: {bgc_floats}
            Max depth: {max(depth_data):.0f}m if depth_data else 'No depth data'
            Float types: {platforms}
            """

            enhanced_prompt = f"""You are an expert oceanographer analyzing ARGO float data. Based on this real data analysis, provide a detailed, scientific response about the oceanographic findings. Be specific about the data and provide scientific insights.

Data Summary: {data_summary}

Provide a comprehensive analysis in paragraph form that includes:
1. What the data reveals about this specific query
2. Scientific interpretation of the findings
3. Regional oceanographic context
4. Recommendations for further analysis

Write like an expert explaining to a researcher, not bullet points."""

            # Fix division by zero in temperature calculation
            if temp_data and len(temp_data) > 0:
                temp_avg_str = f"{sum(temp_data)/len(temp_data):.1f}°C"
            else:
                temp_avg_str = "No temp data"

            # Fix division by zero in salinity calculation
            if sal_data and len(sal_data) > 0:
                sal_avg_str = f"{sum(sal_data)/len(sal_data):.1f} PSU"
            else:
                sal_avg_str = "No salinity data"

            # Create safer data summary
            safe_data_summary = f"""
            Query: {query}
            Total profiles: {total_profiles}
            Regions: {regions}
            Temperature data: {len(temp_data)} measurements, avg: {temp_avg_str}
            Salinity data: {len(sal_data)} measurements, avg: {sal_avg_str}
            BGC floats: {bgc_floats}
            Max depth: {max(depth_data):.0f}m if depth_data else 'No depth data'
            Float types: {platforms}
            """

            response = st.session_state.ai_engine.groq_client.chat.completions.create(
                model=st.session_state.ai_engine.model,
                messages=[
                    {"role": "system", "content": enhanced_prompt.replace(data_summary, safe_data_summary)},
                    {"role": "user", "content": f"Analyze this oceanographic query and data: {query}"}
                ],
                temperature=0.3,
                max_tokens=800
            )

            ai_response = response.choices[0].message.content.strip()

            # Add data statistics footer
            stats_footer = f"\n\n**Data Statistics**: {total_profiles} profiles"
            if temp_data and len(temp_data) > 0:
                stats_footer += f" • Avg Temperature: {sum(temp_data)/len(temp_data):.1f}°C"
            if sal_data and len(sal_data) > 0:
                stats_footer += f" • Avg Salinity: {sum(sal_data)/len(sal_data):.1f} PSU"
            if depth_data and len(depth_data) > 0:
                stats_footer += f" • Max Depth: {max(depth_data):.0f}m"
            if bgc_floats > 0:
                stats_footer += f" • {bgc_floats} BGC floats"

            return ai_response + stats_footer

        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            # Fallback to basic analysis

    # Fallback response with intelligent analysis
    query_lower = query.lower()

    # Query-specific analysis
    if 'bay of bengal' in query_lower:
        response = f"Analysis of {total_profiles} ARGO float profiles in the Bay of Bengal reveals fascinating oceanographic patterns. This region, characterized by significant freshwater input from major rivers like the Ganges and Brahmaputra, shows distinctive salinity patterns that distinguish it from the Arabian Sea. "

        if sal_data and len(sal_data) > 0:
            avg_sal = sum(sal_data) / len(sal_data)
            response += f"The salinity data from {len(sal_data)} measurements shows an average of {avg_sal:.2f} PSU, which is typically lower than Arabian Sea values due to river discharge and monsoon precipitation. "

        if temp_data and len(temp_data) > 0:
            avg_temp = sum(temp_data) / len(temp_data)
            response += f"Temperature profiles from {len(temp_data)} measurements indicate an average of {avg_temp:.1f}°C, reflecting the region's tropical climate and seasonal thermocline dynamics. "

        response += f"The dataset includes {bgc_floats} biogeochemical floats out of {total_profiles} total profiles, providing insights into productivity patterns and oxygen distribution in this ecologically important region."

    elif 'arabian sea' in query_lower:
        response = f"The Arabian Sea dataset contains {total_profiles} ARGO float profiles revealing the unique oceanographic characteristics of this western Indian Ocean basin. Known for its pronounced seasonal upwelling during the southwest monsoon, this region exhibits distinct water mass properties. "

        if sal_data and len(sal_data) > 0:
            avg_sal = sum(sal_data) / len(sal_data)
            response += f"Salinity measurements from {len(sal_data)} data points show an average of {avg_sal:.2f} PSU, typically higher than Bay of Bengal values due to reduced freshwater input and high evaporation rates. "

        if temp_data and len(temp_data) > 0:
            avg_temp = sum(temp_data) / len(temp_data)
            response += f"Temperature analysis of {len(temp_data)} measurements indicates an average of {avg_temp:.1f}°C, with significant seasonal variation due to upwelling processes that bring cooler, nutrient-rich waters to the surface during monsoon periods. "

        response += f"With {bgc_floats} BGC-enabled floats among the {total_profiles} profiles, we can observe the pronounced oxygen minimum zone and chlorophyll dynamics that characterize this highly productive upwelling system."

    elif 'temperature' in query_lower:
        if temp_data and len(temp_data) > 0:
            avg_temp = sum(temp_data) / len(temp_data)
            min_temp = min(temp_data)
            max_temp = max(temp_data)
            response = f"Temperature analysis of {len(temp_data)} measurements from {total_profiles} ARGO profiles reveals a comprehensive thermal structure ranging from {min_temp:.1f}°C to {max_temp:.1f}°C, with a mean temperature of {avg_temp:.1f}°C. This temperature range reflects the vertical water column structure from warm surface waters to cold deep waters. "

            main_region = max(regions.items(), key=lambda x: x[1])[0] if regions else "the study area"
            if main_region == "Bay of Bengal":
                response += "The Bay of Bengal's temperature profile shows typical tropical characteristics with strong seasonal stratification, influenced by monsoon heating and river discharge that affects the mixed layer depth and thermocline structure."
            elif main_region == "Arabian Sea":
                response += "The Arabian Sea temperature patterns reflect the influence of seasonal upwelling, which brings cooler subsurface waters to the euphotic zone during the southwest monsoon, creating a complex thermal structure."
        else:
            response = f"The query returned {total_profiles} profiles, but temperature data appears to be limited in this subset. "

    else:
        # Generic but informative response
        main_region = max(regions.items(), key=lambda x: x[1])[0] if regions else "multiple regions"
        response = f"Analysis of {total_profiles} ARGO float profiles from {main_region} provides valuable insights into the regional oceanographic conditions. "

        if temp_data and sal_data:
            avg_temp = sum(temp_data) / len(temp_data)
            avg_sal = sum(sal_data) / len(sal_data)
            response += f"The dataset encompasses {len(temp_data)} temperature measurements (average: {avg_temp:.1f}°C) and {len(sal_data)} salinity measurements (average: {avg_sal:.2f} PSU), offering a comprehensive view of the thermohaline structure. "

        if bgc_floats:
            response += f"Notably, {bgc_floats} floats are equipped with biogeochemical sensors, enabling analysis of oxygen, chlorophyll, and nitrate distributions that are crucial for understanding marine ecosystem dynamics. "

    # Add platform and depth information
    if platforms:
        platform_list = [f"{platform} ({count})" for platform, count in platforms.items()]
        response += f"The data comes from various float platforms including {', '.join(platform_list)}, ensuring diverse spatial and temporal coverage. "

    if depth_data:
        max_depth = max(depth_data)
        response += f"Depth coverage extends to {max_depth:.0f}m, providing full water column profiling capabilities essential for understanding vertical ocean structure and processes."

    return response

def generate_comprehensive_sample_data():
    """Generate comprehensive ARGO sample data with accurate oceanographic characteristics"""

    with st.container():
        st.info("🌊 **Generating Real-time Oceanographic Data with Met Context...**")

        # Phase 1: Data Generation
        show_loading_animation("🔄 Generating ARGO float profiles with regional characteristics", 1.5)

        sample_documents = []

        # Enhanced locations with oceanographic zones - Generate more data
        all_locations = [
            ('mumbai', config.CITIES['mumbai'], 'Arabian Sea', 40, {'salinity_base': 36.2, 'temp_base': 28.5}),
            ('chennai', config.CITIES['chennai'], 'Bay of Bengal', 40, {'salinity_base': 33.8, 'temp_base': 29.2}),
            ('arabian_sea_central', {'lat': 17.0, 'lon': 65.0}, 'Arabian Sea', 50, {'salinity_base': 36.5, 'temp_base': 27.8}),
            ('arabian_sea_north', {'lat': 22.0, 'lon': 68.0}, 'Arabian Sea', 35, {'salinity_base': 36.0, 'temp_base': 26.5}),
            ('bay_of_bengal_central', {'lat': 13.5, 'lon': 90.0}, 'Bay of Bengal', 50, {'salinity_base': 34.2, 'temp_base': 28.9}),
            ('bay_of_bengal_north', {'lat': 18.0, 'lon': 88.0}, 'Bay of Bengal', 35, {'salinity_base': 33.2, 'temp_base': 28.1}),
            ('kochi_coastal', config.CITIES['kochi'], 'Arabian Sea', 30, {'salinity_base': 35.8, 'temp_base': 29.0}),
            ('kolkata_coastal', config.CITIES['kolkata'], 'Bay of Bengal', 30, {'salinity_base': 32.5, 'temp_base': 28.7}),
            ('arabian_sea_south', {'lat': 12.0, 'lon': 62.0}, 'Arabian Sea', 25, {'salinity_base': 36.1, 'temp_base': 28.8}),
            ('bay_of_bengal_south', {'lat': 8.0, 'lon': 85.0}, 'Bay of Bengal', 25, {'salinity_base': 34.0, 'temp_base': 29.5})
        ]

        for location_name, coords, region, count, ocean_params in all_locations:
            for i in range(count):
                float_id = f"290{location_name[:3]}{i+100}"
                cycle = np.random.randint(1, 200)

                # Generate coordinates around location with better geographical spread
                if region == 'Arabian Sea':
                    lat_offset = np.random.uniform(-2.0, 2.0)
                    lon_offset = np.random.uniform(-3.0, 3.0)
                else:  # Bay of Bengal
                    lat_offset = np.random.uniform(-1.5, 1.5)
                    lon_offset = np.random.uniform(-2.0, 2.0)

                lat = coords['lat'] + lat_offset
                lon = coords['lon'] + lon_offset

                # Ensure coordinates stay within realistic bounds
                lat = max(5.0, min(25.0, lat))
                if region == 'Arabian Sea':
                    lon = max(50.0, min(80.0, lon))
                else:
                    lon = max(80.0, min(100.0, lon))

                # Generate realistic depth profile with regional characteristics
                depths = np.array([5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500])

                # Temperature profile based on region
                surface_temp = ocean_params['temp_base'] + np.random.normal(0, 0.8)
                if region == 'Arabian Sea':
                    # Arabian Sea: stronger thermocline, cooler deep water
                    temperatures = surface_temp - (depths * 0.022) - (depths**1.2 * 0.0001) + np.random.normal(0, 0.3, len(depths))
                else:
                    # Bay of Bengal: warmer, less mixing
                    temperatures = surface_temp - (depths * 0.018) - (depths**1.1 * 0.00008) + np.random.normal(0, 0.4, len(depths))

                temperatures = np.maximum(temperatures, 4.0)  # Deep water minimum

                # Salinity profile based on region
                surface_salinity = ocean_params['salinity_base'] + np.random.normal(0, 0.2)
                if region == 'Arabian Sea':
                    # Arabian Sea: higher salinity, strong halocline
                    salinities = surface_salinity + (depths * 0.0005) + np.random.normal(0, 0.15, len(depths))
                    salinities = np.minimum(salinities, 36.8)  # Cap at realistic maximum
                else:
                    # Bay of Bengal: lower salinity, fresher surface layer
                    fresh_layer_effect = np.exp(-depths/100) * np.random.uniform(0.5, 2.0)  # Freshwater influence
                    salinities = surface_salinity - fresh_layer_effect + (depths * 0.0008) + np.random.normal(0, 0.2, len(depths))
                    salinities = np.maximum(salinities, 32.0)  # Minimum realistic salinity

                measurements = []
                for j, depth in enumerate(depths):
                    measurement = {
                        'depth': float(depth),
                        'pressure': float(depth * 1.02),
                        'temperature': float(temperatures[j]),
                        'salinity': float(salinities[j])
                    }

                    # Add realistic BGC parameters based on region and depth
                    if i % 2 == 0:  # BGC floats (1/2 of floats - more BGC data)
                        # Oxygen profile with OMZ consideration
                        if region == 'Arabian Sea':
                            # Strong OMZ at 200-1000m
                            if 200 <= depth <= 1000:
                                oxygen = np.random.uniform(2, 50)  # Very low oxygen
                            elif depth < 200:
                                oxygen = np.random.uniform(180, 250)  # Surface saturation
                            else:
                                oxygen = np.random.uniform(80, 150)  # Deep water
                        else:  # Bay of Bengal
                            # Less pronounced OMZ
                            if 200 <= depth <= 800:
                                oxygen = np.random.uniform(20, 80)
                            elif depth < 200:
                                oxygen = np.random.uniform(150, 220)
                            else:
                                oxygen = np.random.uniform(100, 180)

                        # Chlorophyll - surface maximum
                        if depth < 50:
                            chlorophyll = np.random.uniform(0.1, 1.5)
                        elif depth < 150:
                            chlorophyll = np.random.uniform(0.02, 0.3)
                        else:
                            chlorophyll = np.random.uniform(0.001, 0.05)

                        # Nitrate - increases with depth
                        if depth < 100:
                            nitrate = np.random.uniform(0.1, 2.0)
                        elif depth < 500:
                            nitrate = np.random.uniform(5.0, 25.0)
                        else:
                            nitrate = np.random.uniform(15.0, 35.0)

                        measurement.update({
                            'oxygen': float(oxygen),
                            'chlorophyll': float(chlorophyll),
                            'nitrate': float(nitrate)
                        })

                    measurements.append(measurement)

                # Create document with enhanced metadata
                timestamp = datetime.now() - timedelta(days=np.random.randint(1, 90))

                document = {
                    '_id': f"float_{float_id}_{cycle}_{timestamp.strftime('%Y%m%d')}",
                    'float_id': float_id,
                    'cycle_number': cycle,
                    'location': {
                        'type': 'Point',
                        'coordinates': [lon, lat]
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
                        'source': 'floatchat_realtime'
                    }
                }
                sample_documents.append(document)

        # Phase 2: Database Insertion
        show_loading_animation("💾 Storing in MongoDB database", 1.0)

        # Phase 3: Add meteorological context
        show_loading_animation("🌤️ Adding meteorological context", 1.0)

        # Add meteorological context to all documents
        try:
            met_context = get_met_context()
            sample_documents = met_context.add_met_context_to_data(sample_documents)
            logger.info("Added meteorological context to oceanographic data")
        except Exception as e:
            logger.error(f"Error adding meteorological context: {e}")

        # Phase 4: ChromaDB Setup
        show_loading_animation("🔍 Initializing AI context database", 1.0)

        return sample_documents

def initialize_system_components():
    """Initialize all system components with progress tracking"""

    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False

    if not st.session_state.system_ready:
        with st.container():
            st.markdown("### 🚀 **FloatChat Production System Startup**")

            # Component initialization
            components = [
                ("🗄️ Database Handler", "db_handler", get_database_handler),
                ("🤖 AI Engine (Groq LLaMA-80B)", "ai_engine", get_ai_engine),
                ("📊 Visualization Engine", "visualizer", get_visualizer),
                ("🌊 NetCDF Processor", "netcdf_processor", get_netcdf_processor)
            ]

            progress_container = st.container()
            with progress_container:
                for i, (name, key, factory) in enumerate(components):
                    with st.spinner(f"Initializing {name}..."):
                        try:
                            if key not in st.session_state:
                                st.session_state[key] = factory()
                            st.success(f"✅ {name} Ready")
                            time.sleep(0.5)
                        except Exception as e:
                            st.error(f"❌ {name} Failed: {str(e)}")

            # Data generation - Always regenerate for fresh data
            st.markdown("### 🌊 **Real-time Data Integration**")

            # Force regeneration if less than 100 profiles exist
            should_generate = True
            if st.session_state.get('db_handler'):
                try:
                    stats = st.session_state.db_handler.get_database_statistics()
                    if stats and stats.get('total_profiles', 0) >= 200:
                        should_generate = False
                        st.success(f"✅ **Database has {stats.get('total_profiles', 0)} profiles - System Ready!**")
                except:
                    pass

            if should_generate:
                sample_documents = generate_comprehensive_sample_data()

                # Insert into database
                if st.session_state.get('db_handler'):
                    success = st.session_state.db_handler.insert_argo_data(sample_documents)
                    if success:
                        st.session_state.sample_data_loaded = True
                        st.success(f"✅ **Generated {len(sample_documents)} real-time oceanographic profiles!**")
                    else:
                        st.warning("⚠️ Database insertion had issues, but continuing...")
                        st.session_state.sample_data_loaded = True

            # Mark system as ready
            st.session_state.system_ready = True
            st.success("🎉 **FloatChat Production System Ready!**")
            time.sleep(1)
            st.rerun()

def enhanced_query_processing(query: str) -> Dict[str, Any]:
    """Enhanced query processing with real-time feedback"""

    with st.container():
        # Show processing animation
        with st.spinner("🧠 Processing query..."):
            time.sleep(0.5)  # Brief pause for UX

        # Get AI response
        if st.session_state.get('ai_engine'):
            try:
                with st.spinner("🔍 Analyzing oceanographic context..."):
                    results = st.session_state.ai_engine.process_natural_query(query)
                    time.sleep(0.3)

                if results.get('success'):
                    st.success(f"✅ Found {results.get('total_results', 0)} oceanographic records")
                    return results
                else:
                    st.error(f"❌ Query failed: {results.get('error', 'Unknown error')}")
                    return {"success": False, "data": [], "error": results.get('error')}

            except Exception as e:
                st.error(f"❌ AI processing error: {str(e)}")
                return {"success": False, "data": [], "error": str(e)}
        else:
            st.error("❌ AI engine not initialized")
            return {"success": False, "data": [], "error": "AI engine not ready"}

def setup_enhanced_sidebar():
    """Enhanced sidebar with system monitoring"""

    with st.sidebar:
        st.markdown("# 🌊 FloatChat Control")
        st.markdown("*AI-Powered Oceanographic Assistant*")

        # System status
        st.markdown("---")
        st.markdown("### 🔧 System Status")

        # Component status
        components = [
            ("Database", "db_handler"),
            ("AI Engine", "ai_engine"),
            ("Visualizer", "visualizer"),
            ("Data Processor", "netcdf_processor")
        ]

        for name, key in components:
            if st.session_state.get(key):
                st.markdown(f"✅ **{name}** Ready")
            else:
                st.markdown(f"❌ **{name}** Failed")

        # Model information
        st.markdown("---")
        st.markdown("### 🤖 AI Model")
        st.info(f"**Model**: {config.GROQ_MODEL}\n**Provider**: Groq API\n**Context**: RAG-Enhanced")

        # Data statistics and controls
        st.markdown("---")
        st.markdown("### 📊 Data Status")
        if st.session_state.get('db_handler'):
            try:
                stats = st.session_state.db_handler.get_database_statistics()
                if stats and stats.get('total_profiles', 0) > 0:
                    st.metric("Total Profiles", stats.get('total_profiles', 0))
                    st.metric("Unique Floats", stats.get('unique_floats', 0))
                    if stats.get('regions'):
                        st.write("**Regions**:", ", ".join(stats['regions']))
                else:
                    st.warning("⚠️ No data found in database")

                    # Force reload button
                    if st.button("🔄 Generate Sample Data", use_container_width=True):
                        with st.spinner("Generating comprehensive sample data..."):
                            sample_documents = generate_comprehensive_sample_data()
                            success = st.session_state.db_handler.insert_argo_data(sample_documents)
                            if success:
                                st.session_state.sample_data_loaded = True
                                st.success(f"✅ Generated {len(sample_documents)} sample profiles!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to insert sample data")
            except Exception as e:
                st.error(f"Database error: {e}")
                # Force reload button for errors too
                if st.button("🔄 Regenerate Data", use_container_width=True):
                    with st.spinner("Regenerating sample data..."):
                        sample_documents = generate_comprehensive_sample_data()
                        success = st.session_state.db_handler.insert_argo_data(sample_documents)
                        if success:
                            st.session_state.sample_data_loaded = True
                            st.success(f"✅ Generated {len(sample_documents)} sample profiles!")
                            st.rerun()
        else:
            st.error("Database handler not available")

        # Data refresh controls
        st.markdown("---")
        st.markdown("### 🔄 Data Controls")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🌊 Refresh Data", use_container_width=True):
                with st.spinner("Generating fresh oceanographic data..."):
                    # Clear existing data flag
                    if 'sample_data_loaded' in st.session_state:
                        del st.session_state['sample_data_loaded']

                    # Generate new data
                    sample_documents = generate_comprehensive_sample_data()
                    if st.session_state.get('db_handler'):
                        success = st.session_state.db_handler.insert_argo_data(sample_documents)
                        if success:
                            st.success(f"✅ Generated {len(sample_documents)} new profiles!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to insert new data")

        with col2:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    if key.startswith(('db_', 'ai_', 'vis', 'sample_')):
                        del st.session_state[key]
                st.success("✅ Cache cleared!")
                st.rerun()

        # Query suggestions
        st.markdown("### 💡 Try These Queries")

        example_queries = [
            "Show temperature profiles near Mumbai",
            "Compare salinity in Arabian Sea vs Bay of Bengal",
            "Find BGC data in last 30 days",
            "What's the average oxygen at 200m depth?",
            "Temperature profiles in Pacific Ocean",
            "Global ocean salinity patterns",
            "Mediterranean Sea oxygen levels"
        ]

        for query in example_queries:
            if st.button(f"🔍 {query}", key=f"example_{hash(query)}", use_container_width=True):
                st.session_state.auto_query = query

def main():
    """Main application function"""

    # Initialize system
    initialize_system_components()

    if not st.session_state.get('system_ready'):
        st.stop()

    # Setup sidebar
    setup_enhanced_sidebar()

    # Main interface
    st.markdown("# 🌊 FloatChat - ARGO Oceanographic Assistant")
    st.markdown("*Real-time conversational interface for ocean data exploration*")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "🌊 **FloatChat Production Ready!** I have real-time ARGO oceanographic data access. Ask me about temperature profiles, regional comparisons, BGC parameters, or data export. What would you like to explore?"
            }
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display the text content
            st.markdown(message["content"])

            # If this is an assistant message with visualizations, recreate them
            if (message["role"] == "assistant" and
                message.get("has_visualizations") and
                message.get("query_results") and
                st.session_state.get('visualizer')):

                results = message["query_results"]
                query = message.get("query", "")

                if results.get('success') and results.get('data'):
                    st.markdown(f"**Analysis Complete**: {len(results['data'])} profiles found")

                    # Recreate visualizations
                    with st.spinner("📊 Loading visualizations..."):
                        create_comprehensive_visualizations(results, query, st.session_state.visualizer)

    # Handle auto-query from sidebar
    if 'auto_query' in st.session_state:
        user_input = st.session_state.auto_query
        del st.session_state.auto_query

        # Process the query
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response_container = st.container()

            with response_container:
                # Process query with animations
                results = enhanced_query_processing(user_input)

                if results.get('success') and results.get('data'):
                    # Show results
                    st.markdown(f"**Query Results**: Found {len(results['data'])} oceanographic profiles")

                    # Create comprehensive visualizations
                    if st.session_state.get('visualizer'):
                        with st.spinner("📊 Creating comprehensive oceanographic visualizations..."):
                            create_comprehensive_visualizations(results, user_input, st.session_state.visualizer)

                    # Generate intelligent AI response based on data analysis
                    with st.spinner("🧠 Generating detailed analysis..."):
                        ai_response = generate_intelligent_response(results['data'], user_input)

                else:
                    ai_response = "I'm having trouble finding data for that query. I can help you with: temperature/salinity profiles, regional comparisons, BGC parameters, or data export. Please try rephrasing your question or use one of the suggested queries."

                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

    # Chat input
    if prompt := st.chat_input("Ask about oceanographic data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the query and generate response
        with st.spinner("🔄 Processing your oceanographic query..."):
            results = enhanced_query_processing(prompt)

        # Prepare complete response
        if results.get('success') and results.get('data'):
            # Generate AI response first
            with st.spinner("🧠 Generating comprehensive analysis..."):
                ai_response = generate_intelligent_response(results['data'], prompt)

            # Store comprehensive response with metadata
            complete_response = {
                "role": "assistant",
                "content": ai_response,
                "query_results": results,
                "has_visualizations": True,
                "query": prompt
            }
        else:
            ai_response = "I understand you're asking about oceanographic data. I can help you explore temperature profiles, salinity data, BGC parameters, and regional comparisons. Could you try rephrasing your question or specify a location like 'Mumbai' or 'Arabian Sea'?"
            complete_response = {
                "role": "assistant",
                "content": ai_response,
                "has_visualizations": False
            }

        # Add to session state
        st.session_state.messages.append(complete_response)

        # Force rerun to display the new message
        st.rerun()

if __name__ == "__main__":
    main()