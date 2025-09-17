"""
Meteorological Data Integration
Simple meteorological context for oceanographic analysis
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeteorologicalContext:
    """Simple meteorological context generator for oceanographic data"""

    def __init__(self):
        # Seasonal patterns for different regions
        self.seasonal_patterns = {
            'Arabian Sea': {
                'monsoon_months': [6, 7, 8, 9],  # June to September
                'post_monsoon': [10, 11, 12],
                'pre_monsoon': [3, 4, 5],
                'winter': [1, 2],
                'characteristics': {
                    'monsoon': {'wind_speed': (12, 20), 'wave_height': (2.5, 4.0), 'precipitation': (800, 1200)},
                    'post_monsoon': {'wind_speed': (8, 12), 'wave_height': (1.5, 2.5), 'precipitation': (50, 150)},
                    'pre_monsoon': {'wind_speed': (6, 10), 'wave_height': (1.0, 2.0), 'precipitation': (20, 80)},
                    'winter': {'wind_speed': (4, 8), 'wave_height': (0.8, 1.5), 'precipitation': (10, 50)}
                }
            },
            'Bay of Bengal': {
                'monsoon_months': [6, 7, 8, 9],
                'post_monsoon': [10, 11, 12],
                'pre_monsoon': [3, 4, 5],
                'winter': [1, 2],
                'characteristics': {
                    'monsoon': {'wind_speed': (10, 18), 'wave_height': (2.0, 3.5), 'precipitation': (1000, 1500)},
                    'post_monsoon': {'wind_speed': (6, 12), 'wave_height': (1.2, 2.5), 'precipitation': (100, 300)},
                    'pre_monsoon': {'wind_speed': (5, 9), 'wave_height': (0.8, 1.8), 'precipitation': (30, 100)},
                    'winter': {'wind_speed': (3, 7), 'wave_height': (0.6, 1.2), 'precipitation': (15, 60)}
                }
            }
        }

    def get_seasonal_context(self, timestamp: datetime, region: str) -> Dict[str, Any]:
        """Get seasonal meteorological context for a given timestamp and region"""
        try:
            if region not in self.seasonal_patterns:
                return {}

            month = timestamp.month
            patterns = self.seasonal_patterns[region]

            # Determine season
            if month in patterns['monsoon_months']:
                season = 'monsoon'
                season_name = 'Southwest Monsoon'
            elif month in patterns['post_monsoon']:
                season = 'post_monsoon'
                season_name = 'Post-Monsoon'
            elif month in patterns['pre_monsoon']:
                season = 'pre_monsoon'
                season_name = 'Pre-Monsoon'
            else:
                season = 'winter'
                season_name = 'Winter'

            characteristics = patterns['characteristics'][season]

            # Generate realistic values within seasonal ranges
            wind_speed = np.random.uniform(*characteristics['wind_speed'])
            wave_height = np.random.uniform(*characteristics['wave_height'])
            precipitation = np.random.uniform(*characteristics['precipitation'])

            return {
                'season': season_name,
                'month': timestamp.strftime('%B'),
                'wind_speed_ms': round(wind_speed, 1),
                'significant_wave_height_m': round(wave_height, 1),
                'monthly_precipitation_mm': round(precipitation, 0),
                'oceanographic_impact': self._get_oceanographic_impact(season, region)
            }

        except Exception as e:
            logger.error(f"Error getting seasonal context: {e}")
            return {}

    def _get_oceanographic_impact(self, season: str, region: str) -> str:
        """Get oceanographic impact description for season and region"""
        impacts = {
            'Arabian Sea': {
                'monsoon': 'Strong upwelling along western coast, enhanced mixing, increased productivity',
                'post_monsoon': 'Stratification development, surface warming, reduced mixing',
                'pre_monsoon': 'Stable stratification, warm surface layer, minimal mixing',
                'winter': 'Deep convection, uniform mixing, cooler surface temperatures'
            },
            'Bay of Bengal': {
                'monsoon': 'Heavy freshwater input, strong stratification, cyclone activity',
                'post_monsoon': 'Residual stratification, cyclone season continues, river discharge',
                'pre_monsoon': 'Surface warming, stable conditions, minimal river input',
                'winter': 'Moderate mixing, northeast monsoon effects, cooler temperatures'
            }
        }

        return impacts.get(region, {}).get(season, 'Normal seasonal oceanographic conditions')

    def get_cyclone_context(self, timestamp: datetime, latitude: float, longitude: float) -> Dict[str, Any]:
        """Get cyclone season context"""
        month = timestamp.month

        # Bay of Bengal cyclone seasons
        if 80 <= longitude <= 100 and 5 <= latitude <= 25:
            if month in [10, 11, 12]:  # Post-monsoon cyclone season
                return {
                    'cyclone_season': 'Active (Post-monsoon)',
                    'risk_level': 'High',
                    'typical_tracks': 'Northwest towards Odisha/Andhra Pradesh coasts',
                    'oceanographic_effect': 'Deep mixing, surface cooling, enhanced nutrients'
                }
            elif month in [4, 5, 6]:  # Pre-monsoon cyclone season
                return {
                    'cyclone_season': 'Moderate (Pre-monsoon)',
                    'risk_level': 'Medium',
                    'typical_tracks': 'Northward along eastern Indian coast',
                    'oceanographic_effect': 'Mixed layer deepening, upwelling enhancement'
                }

        # Arabian Sea cyclone context
        elif 50 <= longitude <= 80 and 5 <= latitude <= 25:
            if month in [5, 6]:  # Pre-monsoon
                return {
                    'cyclone_season': 'Active (Pre-monsoon)',
                    'risk_level': 'Medium',
                    'typical_tracks': 'Northward along western coast',
                    'oceanographic_effect': 'Coastal upwelling enhancement, mixing'
                }
            elif month in [10, 11]:  # Post-monsoon
                return {
                    'cyclone_season': 'Low Activity',
                    'risk_level': 'Low',
                    'typical_tracks': 'Westward towards Arabian Peninsula',
                    'oceanographic_effect': 'Minimal direct impact on Indian waters'
                }

        return {
            'cyclone_season': 'Inactive',
            'risk_level': 'Low',
            'oceanographic_effect': 'Normal circulation patterns'
        }

    def add_met_context_to_data(self, oceanographic_data: List[Dict]) -> List[Dict]:
        """Add meteorological context to oceanographic data"""
        enhanced_data = []

        for record in oceanographic_data:
            enhanced_record = record.copy()

            try:
                timestamp = record.get('timestamp')
                region = record.get('region', '')
                location = record.get('location', {})

                if timestamp and location and 'coordinates' in location:
                    lon, lat = location['coordinates']

                    # Add seasonal context
                    seasonal_context = self.get_seasonal_context(timestamp, region)
                    if seasonal_context:
                        enhanced_record['meteorological_context'] = seasonal_context

                    # Add cyclone context
                    cyclone_context = self.get_cyclone_context(timestamp, lat, lon)
                    if cyclone_context:
                        enhanced_record['cyclone_context'] = cyclone_context

                    # Add general environmental context
                    enhanced_record['environmental_summary'] = self._generate_environmental_summary(
                        seasonal_context, cyclone_context, region
                    )

            except Exception as e:
                logger.error(f"Error adding met context to record: {e}")

            enhanced_data.append(enhanced_record)

        return enhanced_data

    def _generate_environmental_summary(self, seasonal: Dict, cyclone: Dict, region: str) -> str:
        """Generate environmental summary for the data"""
        try:
            summary_parts = []

            if seasonal:
                summary_parts.append(f"Season: {seasonal.get('season', 'Unknown')}")
                summary_parts.append(f"Wind: {seasonal.get('wind_speed_ms', 'N/A')} m/s")
                summary_parts.append(f"Wave Height: {seasonal.get('significant_wave_height_m', 'N/A')} m")

            if cyclone:
                summary_parts.append(f"Cyclone Risk: {cyclone.get('risk_level', 'Unknown')}")

            if region:
                summary_parts.append(f"Region: {region}")

            return " | ".join(summary_parts) if summary_parts else "Environmental data available"

        except Exception as e:
            logger.error(f"Error generating environmental summary: {e}")
            return "Environmental context analysis available"

def get_met_context():
    """Factory function to get meteorological context instance"""
    return MeteorologicalContext()