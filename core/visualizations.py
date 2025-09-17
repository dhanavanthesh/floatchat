"""
Enhanced Oceanographic Visualizations
Advanced Plotly and Folium visualizations for ARGO data analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins
from config import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class EnhancedOceanographicVisualizer:
    """Advanced visualization system for oceanographic data"""

    def __init__(self):
        # Enhanced color palette for different regions and parameters
        self.color_palette = {
            'Arabian Sea': '#FF6B6B',
            'Bay of Bengal': '#4ECDC4',
            'Indian Ocean': '#45B7D1',
            'Equatorial Pacific': '#96CEB4',
            'temperature': '#FF4B4B',
            'salinity': '#1F77B4',
            'pressure': '#FF7F0E',
            'oxygen': '#2E8B57',
            'chlorophyll': '#32CD32',
            'nitrate': '#8A2BE2'
        }

        # Depth color scale
        self.depth_colorscale = [
            [0.0, '#FF0000'],    # Red - Surface
            [0.2, '#FF8C00'],    # Orange - Shallow
            [0.4, '#FFD700'],    # Gold - Intermediate
            [0.6, '#32CD32'],    # Green - Mid-depth
            [0.8, '#0000FF'],    # Blue - Deep
            [1.0, '#000080']     # Navy - Very deep
        ]

    def create_temperature_depth_profile(self, data: List[Dict], title: str = "Temperature-Depth Profile") -> go.Figure:
        """Create enhanced temperature-depth profile visualization"""
        try:
            if not data:
                return self._create_empty_plot("No data available for temperature profile")

            # Process data
            df_list = []
            for record in data:
                if 'measurements' in record and record['measurements']:
                    float_id = record.get('float_id', 'Unknown')
                    region = record.get('region', 'Unknown')
                    timestamp = record.get('timestamp', datetime.now())
                    platform_type = record.get('platform_type', 'Unknown')

                    for measurement in record['measurements']:
                        if measurement.get('temperature') is not None:
                            df_list.append({
                                'float_id': float_id,
                                'region': region,
                                'timestamp': timestamp,
                                'platform_type': platform_type,
                                'depth': measurement.get('depth', 0),
                                'temperature': measurement.get('temperature'),
                                'salinity': measurement.get('salinity'),
                                'pressure': measurement.get('pressure')
                            })

            if not df_list:
                return self._create_empty_plot("No temperature measurements found")

            df = pd.DataFrame(df_list)

            # Create main figure
            fig = go.Figure()

            # Plot by region with enhanced styling
            for region in df['region'].unique():
                region_data = df[df['region'] == region]
                color = self.color_palette.get(region, '#1f77b4')

                # Add scatter plot with enhanced hover information
                fig.add_trace(go.Scatter(
                    x=region_data['temperature'],
                    y=-region_data['depth'],
                    mode='markers+lines',
                    name=region,
                    marker=dict(
                        color=color,
                        size=6,
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    line=dict(color=color, width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Temperature: %{x:.2f}°C<br>' +
                                'Depth: %{customdata[0]:.1f}m<br>' +
                                'Float ID: %{customdata[1]}<br>' +
                                'Platform: %{customdata[2]}<br>' +
                                'Date: %{customdata[3]}<extra></extra>',
                    customdata=region_data[['depth', 'float_id', 'platform_type', 'timestamp']].values
                ))

            # Enhanced layout with better styling
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'family': 'Arial Black'}
                },
                xaxis_title='Temperature (°C)',
                yaxis_title='Depth (m)',
                template='plotly_white',
                hovermode='closest',
                height=700,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1
                ),
                font=dict(family="Arial", size=12)
            )

            # Add depth reference lines
            fig.add_hline(y=-50, line_dash="dash", line_color="gray", opacity=0.5,
                         annotation_text="Mixed Layer (50m)", annotation_position="right")
            fig.add_hline(y=-200, line_dash="dash", line_color="gray", opacity=0.5,
                         annotation_text="Thermocline (200m)", annotation_position="right")

            fig.update_yaxes(autorange='reversed', gridcolor='lightgray')
            fig.update_xaxes(gridcolor='lightgray')

            return fig

        except Exception as e:
            logger.error(f"Error creating temperature profile: {e}")
            return self._create_empty_plot(f"Error: {str(e)}")

    def create_bgc_multi_parameter_plot(self, data: List[Dict]) -> go.Figure:
        """Create multi-parameter BGC visualization"""
        try:
            if not data:
                return self._create_empty_plot("No BGC data available")

            # Process BGC data
            df_list = []
            for record in data:
                if 'measurements' in record and record['measurements']:
                    float_id = record.get('float_id', 'Unknown')
                    region = record.get('region', 'Unknown')

                    for measurement in record['measurements']:
                        if any(measurement.get(param) is not None for param in ['oxygen', 'chlorophyll', 'nitrate']):
                            df_list.append({
                                'float_id': float_id,
                                'region': region,
                                'depth': measurement.get('depth', 0),
                                'oxygen': measurement.get('oxygen'),
                                'chlorophyll': measurement.get('chlorophyll'),
                                'nitrate': measurement.get('nitrate'),
                                'temperature': measurement.get('temperature'),
                                'salinity': measurement.get('salinity')
                            })

            if not df_list:
                return self._create_empty_plot("No BGC measurements found")

            df = pd.DataFrame(df_list)

            # Create subplots for BGC parameters
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Dissolved Oxygen', 'Chlorophyll-a', 'Nitrate', 'T-S Diagram'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            # Oxygen profile
            for region in df['region'].unique():
                region_data = df[df['region'] == region].dropna(subset=['oxygen'])
                if not region_data.empty:
                    color = self.color_palette.get(region, '#1f77b4')
                    fig.add_trace(
                        go.Scatter(
                            x=region_data['oxygen'], y=-region_data['depth'],
                            mode='markers+lines', name=f'{region} (O₂)',
                            marker=dict(color=color, size=4),
                            line=dict(color=color, width=1.5),
                            showlegend=True
                        ), row=1, col=1
                    )

            # Chlorophyll profile
            for region in df['region'].unique():
                region_data = df[df['region'] == region].dropna(subset=['chlorophyll'])
                if not region_data.empty:
                    color = self.color_palette.get(region, '#1f77b4')
                    fig.add_trace(
                        go.Scatter(
                            x=region_data['chlorophyll'], y=-region_data['depth'],
                            mode='markers+lines', name=f'{region} (Chl)',
                            marker=dict(color=color, size=4),
                            line=dict(color=color, width=1.5, dash='dot'),
                            showlegend=False
                        ), row=1, col=2
                    )

            # Nitrate profile
            for region in df['region'].unique():
                region_data = df[df['region'] == region].dropna(subset=['nitrate'])
                if not region_data.empty:
                    color = self.color_palette.get(region, '#1f77b4')
                    fig.add_trace(
                        go.Scatter(
                            x=region_data['nitrate'], y=-region_data['depth'],
                            mode='markers+lines', name=f'{region} (NO₃)',
                            marker=dict(color=color, size=4),
                            line=dict(color=color, width=1.5, dash='dash'),
                            showlegend=False
                        ), row=2, col=1
                    )

            # T-S Diagram
            for region in df['region'].unique():
                region_data = df[df['region'] == region].dropna(subset=['temperature', 'salinity'])
                if not region_data.empty:
                    color = self.color_palette.get(region, '#1f77b4')
                    fig.add_trace(
                        go.Scatter(
                            x=region_data['salinity'], y=region_data['temperature'],
                            mode='markers', name=f'{region} (T-S)',
                            marker=dict(
                                color=-region_data['depth'],
                                colorscale=self.depth_colorscale,
                                size=6,
                                showscale=True,
                                colorbar=dict(title="Depth (m)", x=1.02)
                            ),
                            showlegend=False
                        ), row=2, col=2
                    )

            # Update axis labels
            fig.update_xaxes(title_text="Oxygen (μmol/kg)", row=1, col=1)
            fig.update_xaxes(title_text="Chlorophyll (mg/m³)", row=1, col=2)
            fig.update_xaxes(title_text="Nitrate (μmol/kg)", row=2, col=1)
            fig.update_xaxes(title_text="Salinity (PSU)", row=2, col=2)

            fig.update_yaxes(title_text="Depth (m)", row=1, col=1, autorange='reversed')
            fig.update_yaxes(title_text="Depth (m)", row=1, col=2, autorange='reversed')
            fig.update_yaxes(title_text="Depth (m)", row=2, col=1, autorange='reversed')
            fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)

            fig.update_layout(
                title="Bio-Geo-Chemical Parameters Analysis",
                template='plotly_white',
                height=800,
                hovermode='closest'
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating BGC plot: {e}")
            return self._create_empty_plot(f"Error: {str(e)}")

    def create_advanced_float_map(self, data: List[Dict], center_lat: float = 15.0, center_lon: float = 75.0) -> folium.Map:
        """Create advanced interactive map with float trajectories"""
        try:
            # Create base map with enhanced styling
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=5,
                tiles=None
            )

            # Add multiple tile layers
            folium.TileLayer('CartoDB positron', name='Light Mode').add_to(m)
            folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)
            folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)

            if not data:
                folium.LayerControl().add_to(m)
                return m

            # Organize data by float for trajectories
            float_data = {}
            for record in data:
                float_id = record.get('float_id', 'Unknown')
                if float_id not in float_data:
                    float_data[float_id] = []
                float_data[float_id].append(record)

            # Create different layers for different regions
            region_groups = {}
            for region in set(record.get('region', 'Unknown') for record in data):
                region_groups[region] = plugins.FeatureGroupSubGroup(
                    folium.FeatureGroup(name=f'{region} Floats').add_to(m),
                    name=region
                )

            # Plot float trajectories and current positions
            for float_id, float_records in float_data.items():
                if not float_records:
                    continue

                # Sort by timestamp for trajectory
                float_records.sort(key=lambda x: x.get('timestamp', datetime.min))

                latest_record = float_records[-1]
                region = latest_record.get('region', 'Unknown')

                # Get trajectory coordinates
                trajectory_coords = []
                for record in float_records:
                    location = record.get('location', {})
                    if location and 'coordinates' in location:
                        coords = location['coordinates']
                        trajectory_coords.append([coords[1], coords[0]])  # [lat, lon]

                # Draw trajectory line
                if len(trajectory_coords) > 1:
                    folium.PolyLine(
                        trajectory_coords,
                        color=self.color_palette.get(region, 'blue'),
                        weight=2,
                        opacity=0.7,
                        popup=f"Float {float_id} trajectory"
                    ).add_to(region_groups.get(region, m))

                # Current position marker
                if trajectory_coords:
                    current_pos = trajectory_coords[-1]

                    # Get latest measurements for popup
                    latest_measurements = latest_record.get('measurements', [])
                    surface_measurement = None
                    if latest_measurements:
                        surface_measurement = min(latest_measurements, key=lambda x: x.get('depth', float('inf')))

                    popup_html = f"""
                    <div style="width:250px;">
                        <h4>🌊 Float {float_id}</h4>
                        <p><b>Region:</b> {region}</p>
                        <p><b>Platform:</b> {latest_record.get('platform_type', 'Unknown')}</p>
                        <p><b>Last Update:</b> {latest_record.get('timestamp', 'Unknown')}</p>
                        <p><b>Coordinates:</b> {current_pos[0]:.3f}°N, {current_pos[1]:.3f}°E</p>
                        <p><b>Total Profiles:</b> {len(float_records)}</p>
                    """

                    if surface_measurement:
                        if surface_measurement.get('temperature'):
                            popup_html += f"<p><b>Surface Temp:</b> {surface_measurement['temperature']:.1f}°C</p>"
                        if surface_measurement.get('salinity'):
                            popup_html += f"<p><b>Surface Salinity:</b> {surface_measurement['salinity']:.2f} PSU</p>"

                    popup_html += "</div>"

                    # Choose marker icon based on platform type
                    platform_type = latest_record.get('platform_type', 'Unknown')
                    if platform_type in ['APEX', 'NOVA']:
                        icon_color = 'red'
                    elif platform_type == 'ARVOR':
                        icon_color = 'blue'
                    else:
                        icon_color = 'green'

                    folium.CircleMarker(
                        location=current_pos,
                        radius=8,
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=f"Float {float_id} - {region}",
                        color='white',
                        fillColor=self.color_palette.get(region, 'blue'),
                        fillOpacity=0.8,
                        weight=2
                    ).add_to(region_groups.get(region, m))

            # Add region boundaries
            self._add_region_boundaries(m)

            # Add measurement density heatmap
            self._add_density_heatmap(m, data)

            # Add layer control
            folium.LayerControl().add_to(m)

            return m

        except Exception as e:
            logger.error(f"Error creating map: {e}")
            return folium.Map(location=[center_lat, center_lon], zoom_start=6)

    def _add_region_boundaries(self, map_obj: folium.Map):
        """Add region boundary overlays"""
        for region_name, region_info in config.REGIONS.items():
            bounds = region_info['bounds']

            # Create rectangle for region boundary
            folium.Rectangle(
                bounds=[[bounds['lat'][0], bounds['lon'][0]],
                       [bounds['lat'][1], bounds['lon'][1]]],
                color=self.color_palette.get(region_name, 'black'),
                fill=False,
                weight=2,
                opacity=0.5,
                popup=f"{region_name} Region"
            ).add_to(map_obj)

    def _add_density_heatmap(self, map_obj: folium.Map, data: List[Dict]):
        """Add measurement density heatmap"""
        heat_data = []
        for record in data:
            location = record.get('location', {})
            if location and 'coordinates' in location:
                coords = location['coordinates']
                heat_data.append([coords[1], coords[0], 1])  # [lat, lon, weight]

        if heat_data:
            plugins.HeatMap(
                heat_data,
                name='Measurement Density',
                radius=15,
                blur=10,
                max_zoom=1
            ).add_to(map_obj)

    def create_salinity_comparison_arabian_bay(self, data: List[Dict]) -> go.Figure:
        """Enhanced salinity comparison between Arabian Sea and Bay of Bengal"""
        try:
            if not data:
                return self._create_empty_plot("No data available for salinity comparison")

            # Separate data by region
            arabian_data = []
            bay_data = []

            for record in data:
                region = record.get('region', '')
                measurements = record.get('measurements', [])

                for measurement in measurements:
                    if measurement.get('salinity') is not None:
                        sal_data = {
                            'depth': measurement.get('depth', 0),
                            'salinity': measurement.get('salinity'),
                            'temperature': measurement.get('temperature'),
                            'float_id': record.get('float_id'),
                            'timestamp': record.get('timestamp'),
                            'lat': record.get('location', {}).get('coordinates', [0, 0])[1],
                            'lon': record.get('location', {}).get('coordinates', [0, 0])[0]
                        }

                        if 'Arabian Sea' in region:
                            arabian_data.append(sal_data)
                        elif 'Bay of Bengal' in region:
                            bay_data.append(sal_data)

            if not arabian_data and not bay_data:
                return self._create_empty_plot("No salinity data found for comparison")

            # Create comprehensive comparison figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Salinity Profiles by Region',
                    'Statistical Comparison',
                    'Salinity vs Temperature',
                    'Geographic Distribution'
                ),
                specs=[[{"type": "scatter"}, {"type": "table"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )

            # 1. Depth profiles comparison
            if arabian_data:
                arabian_df = pd.DataFrame(arabian_data)
                depth_bins = np.arange(0, 2000, 50)
                binned_arabian = []

                for i in range(len(depth_bins)-1):
                    depth_mask = (arabian_df['depth'] >= depth_bins[i]) & (arabian_df['depth'] < depth_bins[i+1])
                    if depth_mask.any():
                        avg_sal = arabian_df[depth_mask]['salinity'].mean()
                        std_sal = arabian_df[depth_mask]['salinity'].std()
                        binned_arabian.append({
                            'depth': depth_bins[i],
                            'salinity': avg_sal,
                            'std': std_sal if not pd.isna(std_sal) else 0
                        })

                if binned_arabian:
                    binned_df = pd.DataFrame(binned_arabian)
                    fig.add_trace(
                        go.Scatter(
                            x=binned_df['salinity'],
                            y=-binned_df['depth'],
                            mode='lines+markers',
                            name='Arabian Sea',
                            line=dict(color='#FF6B6B', width=3),
                            marker=dict(color='#FF6B6B', size=6),
                            error_x=dict(
                                type='data',
                                array=binned_df['std'],
                                visible=True,
                                color='#FF6B6B'
                            ),
                            hovertemplate='<b>Arabian Sea</b><br>Salinity: %{x:.2f} PSU<br>Depth: %{customdata:.0f}m<br>Std Dev: ±%{error_x:.2f}<extra></extra>',
                            customdata=binned_df['depth']
                        ), row=1, col=1
                    )

            if bay_data:
                bay_df = pd.DataFrame(bay_data)
                depth_bins = np.arange(0, 2000, 50)
                binned_bay = []

                for i in range(len(depth_bins)-1):
                    depth_mask = (bay_df['depth'] >= depth_bins[i]) & (bay_df['depth'] < depth_bins[i+1])
                    if depth_mask.any():
                        avg_sal = bay_df[depth_mask]['salinity'].mean()
                        std_sal = bay_df[depth_mask]['salinity'].std()
                        binned_bay.append({
                            'depth': depth_bins[i],
                            'salinity': avg_sal,
                            'std': std_sal if not pd.isna(std_sal) else 0
                        })

                if binned_bay:
                    binned_df = pd.DataFrame(binned_bay)
                    fig.add_trace(
                        go.Scatter(
                            x=binned_df['salinity'],
                            y=-binned_df['depth'],
                            mode='lines+markers',
                            name='Bay of Bengal',
                            line=dict(color='#4ECDC4', width=3),
                            marker=dict(color='#4ECDC4', size=6),
                            error_x=dict(
                                type='data',
                                array=binned_df['std'],
                                visible=True,
                                color='#4ECDC4'
                            ),
                            hovertemplate='<b>Bay of Bengal</b><br>Salinity: %{x:.2f} PSU<br>Depth: %{customdata:.0f}m<br>Std Dev: ±%{error_x:.2f}<extra></extra>',
                            customdata=binned_df['depth']
                        ), row=1, col=1
                    )

            # 2. Statistical comparison table
            stats_data = []
            if arabian_data:
                arabian_salinities = [d['salinity'] for d in arabian_data]
                stats_data.append({
                    'Region': 'Arabian Sea',
                    'Count': len(arabian_salinities),
                    'Mean (PSU)': f"{np.mean(arabian_salinities):.2f}",
                    'Std Dev': f"{np.std(arabian_salinities):.2f}",
                    'Min': f"{np.min(arabian_salinities):.2f}",
                    'Max': f"{np.max(arabian_salinities):.2f}",
                    'Surface Avg': f"{np.mean([s for d in arabian_data for s in [d['salinity']] if d['depth'] < 50]):.2f}" if any(d['depth'] < 50 for d in arabian_data) else "N/A"
                })

            if bay_data:
                bay_salinities = [d['salinity'] for d in bay_data]
                stats_data.append({
                    'Region': 'Bay of Bengal',
                    'Count': len(bay_salinities),
                    'Mean (PSU)': f"{np.mean(bay_salinities):.2f}",
                    'Std Dev': f"{np.std(bay_salinities):.2f}",
                    'Min': f"{np.min(bay_salinities):.2f}",
                    'Max': f"{np.max(bay_salinities):.2f}",
                    'Surface Avg': f"{np.mean([s for d in bay_data for s in [d['salinity']] if d['depth'] < 50]):.2f}" if any(d['depth'] < 50 for d in bay_data) else "N/A"
                })

            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=list(stats_df.columns),
                            fill_color='lightblue',
                            align='center',
                            font=dict(size=12, color='darkblue')
                        ),
                        cells=dict(
                            values=[stats_df[col] for col in stats_df.columns],
                            fill_color='lightgray',
                            align='center',
                            font=dict(size=11)
                        )
                    ), row=1, col=2
                )

            # 3. Salinity vs Temperature scatter
            if arabian_data:
                arabian_df = pd.DataFrame(arabian_data)
                valid_temp_sal = arabian_df.dropna(subset=['temperature', 'salinity'])
                if not valid_temp_sal.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_temp_sal['temperature'],
                            y=valid_temp_sal['salinity'],
                            mode='markers',
                            name='Arabian Sea T-S',
                            marker=dict(
                                color=valid_temp_sal['depth'],
                                colorscale='Reds',
                                size=6,
                                opacity=0.7,
                                colorbar=dict(title="Depth (m)", x=1.05)
                            ),
                            hovertemplate='<b>Arabian Sea</b><br>Temperature: %{x:.1f}°C<br>Salinity: %{y:.2f} PSU<br>Depth: %{marker.color:.0f}m<extra></extra>'
                        ), row=2, col=1
                    )

            if bay_data:
                bay_df = pd.DataFrame(bay_data)
                valid_temp_sal = bay_df.dropna(subset=['temperature', 'salinity'])
                if not valid_temp_sal.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_temp_sal['temperature'],
                            y=valid_temp_sal['salinity'],
                            mode='markers',
                            name='Bay of Bengal T-S',
                            marker=dict(
                                color=valid_temp_sal['depth'],
                                colorscale='Blues',
                                size=6,
                                opacity=0.7
                            ),
                            hovertemplate='<b>Bay of Bengal</b><br>Temperature: %{x:.1f}°C<br>Salinity: %{y:.2f} PSU<br>Depth: %{marker.color:.0f}m<extra></extra>'
                        ), row=2, col=1
                    )

            # 4. Geographic distribution
            if arabian_data:
                arabian_df = pd.DataFrame(arabian_data)
                surface_arabian = arabian_df[arabian_df['depth'] < 50]
                if not surface_arabian.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=surface_arabian['lon'],
                            y=surface_arabian['lat'],
                            mode='markers',
                            name='Arabian Sea Locations',
                            marker=dict(
                                color=surface_arabian['salinity'],
                                colorscale='YlOrRd',
                                size=8,
                                opacity=0.8,
                                colorbar=dict(title="Surface Salinity (PSU)", x=1.1)
                            ),
                            hovertemplate='<b>Arabian Sea</b><br>Lat: %{y:.2f}°<br>Lon: %{x:.2f}°<br>Salinity: %{marker.color:.2f} PSU<extra></extra>'
                        ), row=2, col=2
                    )

            if bay_data:
                bay_df = pd.DataFrame(bay_data)
                surface_bay = bay_df[bay_df['depth'] < 50]
                if not surface_bay.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=surface_bay['lon'],
                            y=surface_bay['lat'],
                            mode='markers',
                            name='Bay of Bengal Locations',
                            marker=dict(
                                color=surface_bay['salinity'],
                                colorscale='YlGnBu',
                                size=8,
                                opacity=0.8
                            ),
                            hovertemplate='<b>Bay of Bengal</b><br>Lat: %{y:.2f}°<br>Lon: %{x:.2f}°<br>Salinity: %{marker.color:.2f} PSU<extra></extra>'
                        ), row=2, col=2
                    )

            # Update layout
            fig.update_xaxes(title_text="Salinity (PSU)", row=1, col=1)
            fig.update_yaxes(title_text="Depth (m)", row=1, col=1, autorange='reversed')

            fig.update_xaxes(title_text="Temperature (°C)", row=2, col=1)
            fig.update_yaxes(title_text="Salinity (PSU)", row=2, col=1)

            fig.update_xaxes(title_text="Longitude", row=2, col=2)
            fig.update_yaxes(title_text="Latitude", row=2, col=2)

            fig.update_layout(
                title={
                    'text': "Comprehensive Salinity Comparison: Arabian Sea vs Bay of Bengal",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'family': 'Arial Black'}
                },
                template='plotly_white',
                height=800,
                hovermode='closest',
                showlegend=True
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating salinity comparison: {e}")
            return self._create_empty_plot(f"Error: {str(e)}")

    def create_regional_comparison(self, data: List[Dict], parameter: str = 'temperature') -> go.Figure:
        """Create advanced regional comparison visualization"""
        try:
            if not data:
                return self._create_empty_plot(f"No data for {parameter} comparison")

            # Process data by region
            region_data = {}
            for record in data:
                region = record.get('region', 'Unknown')
                if region not in region_data:
                    region_data[region] = []

                measurements = record.get('measurements', [])
                for measurement in measurements:
                    if measurement.get(parameter) is not None:
                        region_data[region].append({
                            'depth': measurement.get('depth', 0),
                            'value': measurement.get(parameter),
                            'float_id': record.get('float_id'),
                            'timestamp': record.get('timestamp')
                        })

            if not any(region_data.values()):
                return self._create_empty_plot(f"No {parameter} data found")

            # Create figure with statistical comparison
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    f'{parameter.title()} Profiles',
                    f'{parameter.title()} Statistics',
                    'Depth Distribution',
                    'Regional Box Plot'
                )
            )

            # Profile comparison (subplot 1)
            for region, measurements in region_data.items():
                if measurements:
                    df_region = pd.DataFrame(measurements)
                    color = self.color_palette.get(region, '#1f77b4')

                    # Average profile
                    depth_bins = np.arange(0, 2000, 50)
                    binned_data = []
                    for i in range(len(depth_bins)-1):
                        depth_mask = (df_region['depth'] >= depth_bins[i]) & (df_region['depth'] < depth_bins[i+1])
                        if depth_mask.any():
                            avg_value = df_region[depth_mask]['value'].mean()
                            binned_data.append({'depth': depth_bins[i], 'value': avg_value})

                    if binned_data:
                        binned_df = pd.DataFrame(binned_data)
                        fig.add_trace(
                            go.Scatter(
                                x=binned_df['value'], y=-binned_df['depth'],
                                mode='lines+markers', name=region,
                                line=dict(color=color, width=3),
                                marker=dict(color=color, size=6)
                            ), row=1, col=1
                        )

            # Statistics table (subplot 2)
            stats_data = []
            for region, measurements in region_data.items():
                if measurements:
                    values = [m['value'] for m in measurements]
                    stats_data.append({
                        'Region': region,
                        'Count': len(values),
                        'Mean': np.mean(values),
                        'Std': np.std(values),
                        'Min': np.min(values),
                        'Max': np.max(values)
                    })

            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                fig.add_trace(
                    go.Table(
                        header=dict(values=list(stats_df.columns)),
                        cells=dict(values=[stats_df[col] for col in stats_df.columns])
                    ), row=1, col=2
                )

            # Continue with other subplots...
            fig.update_layout(
                title=f"Regional {parameter.title()} Analysis",
                template='plotly_white',
                height=800
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating regional comparison: {e}")
            return self._create_empty_plot(f"Error: {str(e)}")

    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    def create_depth_profile_plot(self, data: List[Dict]) -> go.Figure:
        """Alias for temperature_depth_profile for compatibility"""
        return self.create_temperature_depth_profile(data)

    def create_ts_diagram(self, data: List[Dict]) -> go.Figure:
        """Create Temperature-Salinity diagram"""
        try:
            temperatures = []
            salinities = []
            depths = []
            regions = []

            for item in data:
                for measurement in item.get('measurements', []):
                    temp = measurement.get('temperature')
                    sal = measurement.get('salinity')
                    depth = measurement.get('depth', 0)
                    region = item.get('region', 'Unknown')

                    if temp is not None and sal is not None:
                        temperatures.append(temp)
                        salinities.append(sal)
                        depths.append(depth)
                        regions.append(region)

            if not temperatures:
                return self._create_empty_plot("No temperature-salinity data available")

            # Create T-S diagram
            fig = go.Figure()

            # Group by region for different colors
            unique_regions = list(set(regions))
            for region in unique_regions:
                region_temps = [t for t, r in zip(temperatures, regions) if r == region]
                region_sals = [s for s, r in zip(salinities, regions) if r == region]
                region_depths = [d for d, r in zip(depths, regions) if r == region]

                color = self.color_palette.get(region, '#1f77b4')

                fig.add_trace(go.Scatter(
                    x=region_sals,
                    y=region_temps,
                    mode='markers',
                    name=region,
                    marker=dict(
                        color=region_depths,
                        colorscale=self.depth_colorscale,
                        size=8,
                        opacity=0.7,
                        colorbar=dict(title="Depth (m)")
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>Salinity: %{x:.2f} PSU<br>Temperature: %{y:.2f}°C<br>Depth: %{marker.color:.0f}m<extra></extra>'
                ))

            fig.update_layout(
                title="Temperature-Salinity Diagram",
                xaxis_title="Salinity (PSU)",
                yaxis_title="Temperature (°C)",
                hovermode='closest',
                showlegend=True
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating T-S diagram: {e}")
            return self._create_empty_plot(f"Error creating T-S diagram: {str(e)}")

    def create_float_trajectory_map(self, data: List[Dict]) -> folium.Map:
        """Alias for advanced_float_map for compatibility"""
        if not data:
            # Create empty map
            return folium.Map(location=[15.0, 75.0], zoom_start=5)

        # Get center from first data point
        first_coords = data[0].get('location', {}).get('coordinates', [75.0, 15.0])
        return self.create_advanced_float_map(data, center_lat=first_coords[1], center_lon=first_coords[0])

def get_visualizer():
    """Factory function to get visualizer instance"""
    return EnhancedOceanographicVisualizer()