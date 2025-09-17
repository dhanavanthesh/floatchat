"""
Advanced AI Engine with RAG Pipeline
Groq LLM integration with Model Context Protocol (MCP) for oceanographic queries
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from groq import Groq
from core.database import UnifiedDatabaseHandler
from core.argo_realtime import get_real_argo_fetcher
from config import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class AdvancedAIEngine:
    """Advanced AI engine with RAG capabilities for oceanographic data queries"""

    def __init__(self):
        self.groq_api_key = config.GROQ_API_KEY
        self.model = config.GROQ_MODEL
        self.max_tokens = config.MAX_TOKENS
        self.temperature = config.TEMPERATURE

        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required")

        self.groq_client = Groq(api_key=self.groq_api_key)
        self.db_handler = UnifiedDatabaseHandler()
        self.real_argo_fetcher = get_real_argo_fetcher()

        # Enhanced system prompt with MCP principles
        self.system_prompt = self.build_system_prompt()

        # Query cache for performance
        self.query_cache = {}
        self.cache_size_limit = 100

    def build_system_prompt(self) -> str:
        """Build comprehensive system prompt for oceanographic queries"""
        return f"""You are an expert oceanographer and MongoDB query specialist with deep knowledge of ARGO float data analysis. Your role is to convert natural language questions about oceanographic data into precise MongoDB aggregation pipelines.

## AVAILABLE DATABASE SCHEMA:

### Main Collection: {config.MONGODB_COLLECTION}
```javascript
{{
  "_id": "float_2901623_20230315",
  "float_id": "2901623",
  "cycle_number": 145,
  "location": {{
    "type": "Point",
    "coordinates": [longitude, latitude]  // [lon, lat] format
  }},
  "timestamp": ISODate("2023-03-15T06:30:00Z"),
  "region": "Arabian Sea" | "Bay of Bengal" | "Indian Ocean",
  "platform_type": "APEX" | "NOVA" | "ARVOR",
  "data_mode": "R" | "A" | "D",  // Real-time, Adjusted, Delayed
  "measurements": [
    {{
      "depth": 10.5,
      "pressure": 11.2,
      "temperature": 28.7,    // Celsius
      "salinity": 36.1,       // PSU
      "oxygen": 245.3,        // μmol/kg (BGC floats)
      "chlorophyll": 0.12,    // mg/m³ (BGC floats)
      "nitrate": 1.8          // μmol/kg (BGC floats)
    }}
  ],
  "quality_flags": {{
    "temperature": "good" | "probably_good" | "probably_bad" | "bad",
    "salinity": "good" | "probably_good" | "probably_bad" | "bad"
  }},
  "metadata": {{
    "institution": "INCOIS",
    "project": "ARGO_INDIA",
    "netcdf_file": "original_file.nc"
  }}
}}
```

## GEOGRAPHIC REGIONS:
- **Arabian Sea**: {config.REGIONS['Arabian Sea']['bounds']}
- **Bay of Bengal**: {config.REGIONS['Bay of Bengal']['bounds']}
- **Indian Ocean**: {config.REGIONS['Indian Ocean']['bounds']}

## MAJOR CITIES COORDINATES:
{json.dumps(config.CITIES, indent=2)}

## OCEANOGRAPHIC PARAMETERS:
- **CTD**: {config.PARAMETERS['CTD']}
- **BGC**: {config.PARAMETERS['BGC']}
- **Derived**: {config.PARAMETERS['DERIVED']}

## QUERY EXAMPLES:

### 1. Geographic Queries:
- "Show floats near Mumbai" → Use $near with Mumbai coordinates
- "Arabian Sea data" → Match by region
- "Equatorial region" → Match latitude range around 0°

### 2. Temporal Queries:
- "Last 3 months" → Date range from 3 months ago to now
- "March 2023" → Specific month range

### 3. Parameter Queries:
- "Temperature profiles" → Focus on temperature measurements
- "Salinity at 100m depth" → Filter measurements by depth range
- "BGC parameters" → Include oxygen, chlorophyll, nitrate

### 4. Complex Queries:
- "Compare regions" → Group by region, compute statistics
- "Depth trends" → Analyze measurements across depth levels

## RESPONSE FORMAT:
Return ONLY a valid MongoDB aggregation pipeline as JSON array. No explanations, no markdown, no additional text.

Examples of correct responses:

For "Show floats in Arabian Sea":
[{{"$match": {{"region": {{"$regex": "Arabian Sea", "$options": "i"}}}}}}]

For "Temperature near Mumbai":
[
  {{"$match": {{"location": {{"$geoWithin": {{"$centerSphere": [[72.8777, 19.0760], 0.031416]}}}}}}}},
  {{"$match": {{"measurements": {{"$elemMatch": {{"temperature": {{"$exists": true, "$ne": null}}}}}}}}}}
]

For "Show me temperature profiles near Mumbai":
[
  {{"$match": {{"location": {{"$geoWithin": {{"$centerSphere": [[72.8777, 19.0760], 0.031416]}}}}}}}}
]

For "profiles near Chennai":
[
  {{"$match": {{"location": {{"$geoWithin": {{"$centerSphere": [[80.2707, 13.0827], 0.031416]}}}}}}}}
]

For "floats around Mumbai":
[
  {{"$match": {{"location": {{"$geoWithin": {{"$centerSphere": [[72.8777, 19.0760], 0.031416]}}}}}}}}
]

For "Salinity at 100m depth":
[
  {{"$unwind": "$measurements"}},
  {{"$match": {{"measurements.depth": {{"$gte": 90, "$lte": 110}}, "measurements.salinity": {{"$exists": true}}}}}},
  {{"$group": {{"_id": "$float_id", "location": {{"$first": "$location"}}, "region": {{"$first": "$region"}}, "measurements": {{"$push": "$measurements"}}}}}}
]

CRITICAL RULES:
1. Always use proper MongoDB syntax
2. MANDATORY: Use $geoWithin with $centerSphere for ALL city/location queries (Mumbai, Chennai, etc.) with coordinates in [longitude, latitude] format
3. For any city mentioned (Mumbai, Chennai, Kolkata, etc.), ALWAYS generate $geoWithin query with 0.031416 radius (approximately 200km)
4. NEVER use $near, $nearSphere, or $geoNear in aggregation pipelines - use $geoWithin instead
5. Use $unwind when analyzing individual measurements
6. Include $exists and null checks for optional parameters
7. For radius in $centerSphere, use 0.031416 (200km in radians)
8. Return only the JSON pipeline, nothing else

CITY COORDINATES TO USE:
- Mumbai: [72.8777, 19.0760]
- Chennai: [80.2707, 13.0827]
- Kolkata: [88.3639, 22.5726]
- Kochi: [76.2673, 9.9312]
- Goa: [73.8278, 15.2993]
- Visakhapatnam: [83.2185, 17.6868]"""

    def get_contextual_information(self, query: str) -> str:
        """Get relevant context from ChromaDB for the query"""
        try:
            context_docs = self.db_handler.search_context(query, n_results=3)
            if context_docs:
                return "\n".join(context_docs)
            return ""
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""

    def extract_location_from_query(self, query: str) -> Optional[Dict[str, float]]:
        """Extract city/location information from natural language query"""
        query_lower = query.lower()

        # Check for known cities with multiple variations
        city_patterns = {
            'mumbai': ['mumbai', 'bombay', 'near mumbai', 'around mumbai'],
            'chennai': ['chennai', 'madras', 'near chennai', 'around chennai'],
            'kolkata': ['kolkata', 'calcutta', 'near kolkata'],
            'kochi': ['kochi', 'cochin', 'near kochi'],
            'goa': ['goa', 'near goa'],
            'visakhapatnam': ['visakhapatnam', 'vizag', 'near vizag']
        }

        for city, patterns in city_patterns.items():
            if city in config.CITIES:
                for pattern in patterns:
                    if pattern in query_lower:
                        coords = config.CITIES[city]
                        return {"latitude": coords['lat'], "longitude": coords['lon'], "city": city}

        # Check for coordinate patterns
        coord_pattern = r'(\d+\.?\d*)[°]?\s*([NS])\s*,?\s*(\d+\.?\d*)[°]?\s*([EW])'
        match = re.search(coord_pattern, query, re.IGNORECASE)
        if match:
            lat_val, lat_dir, lon_val, lon_dir = match.groups()
            lat = float(lat_val) * (1 if lat_dir.upper() == 'N' else -1)
            lon = float(lon_val) * (1 if lon_dir.upper() == 'E' else -1)
            return {"latitude": lat, "longitude": lon}

        return None

    def extract_temporal_from_query(self, query: str) -> Optional[Dict[str, datetime]]:
        """Extract temporal information from query"""
        query_lower = query.lower()
        now = datetime.now()

        # Relative time patterns
        if "last month" in query_lower:
            return {"start_date": now - timedelta(days=30), "end_date": now}
        elif "last 3 months" in query_lower or "last three months" in query_lower:
            return {"start_date": now - timedelta(days=90), "end_date": now}
        elif "last 6 months" in query_lower or "last six months" in query_lower:
            return {"start_date": now - timedelta(days=180), "end_date": now}
        elif "last year" in query_lower:
            return {"start_date": now - timedelta(days=365), "end_date": now}

        # Specific months/years
        month_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
        match = re.search(month_pattern, query_lower)
        if match:
            month_name, year = match.groups()
            month_map = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            month_num = month_map[month_name]
            start_date = datetime(int(year), month_num, 1)
            if month_num == 12:
                end_date = datetime(int(year) + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(int(year), month_num + 1, 1) - timedelta(days=1)
            return {"start_date": start_date, "end_date": end_date}

        return None

    def extract_depth_from_query(self, query: str) -> Optional[Dict[str, float]]:
        """Extract depth information from query"""
        # Pattern for specific depth
        depth_pattern = r'(\d+)\s*m(?:eter)?s?\s*(?:depth|deep)'
        match = re.search(depth_pattern, query.lower())
        if match:
            depth = float(match.group(1))
            return {"target_depth": depth, "tolerance": 10.0}

        # Depth ranges
        range_pattern = r'(\d+)[-–](\d+)\s*m(?:eter)?s?'
        match = re.search(range_pattern, query.lower())
        if match:
            min_depth, max_depth = match.groups()
            return {"min_depth": float(min_depth), "max_depth": float(max_depth)}

        # Surface layer
        if "surface" in query.lower():
            return {"min_depth": 0, "max_depth": 50}

        return None

    def generate_mongodb_pipeline(self, query: str, context: str = "") -> Tuple[List[Dict], str]:
        """Generate MongoDB aggregation pipeline using Groq LLM"""
        try:
            # Build enhanced prompt with context
            enhanced_prompt = self.system_prompt
            if context:
                enhanced_prompt += f"\n\nRELEVANT CONTEXT:\n{context}\n"

            enhanced_prompt += f"\n\nUSER QUERY: {query}\n\nMongoDB Pipeline:"

            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": enhanced_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            pipeline_text = response.choices[0].message.content.strip()

            # Clean and parse the response
            pipeline_text = re.sub(r'```json\s*|\s*```', '', pipeline_text)
            pipeline_text = pipeline_text.strip()

            try:
                pipeline = json.loads(pipeline_text)
                if isinstance(pipeline, dict):
                    pipeline = [pipeline]
                return pipeline, "llm_generated"
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                return [], "json_error"

        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return [], "api_error"


    def process_natural_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query and return results with LLM intelligence"""
        try:
            logger.info(f"Processing global query: {query}")

            # Step 1: LLM analyzes query to extract parameters
            query_analysis = self.analyze_query_with_llm(query)

            # Step 2: Try to get data from database first
            context = self.get_contextual_information(query)
            pipeline, generation_method = self.generate_mongodb_pipeline(query, context)

            results = []
            data_source = "local"

            if pipeline:
                logger.info(f"Generated pipeline: {pipeline}")
                results = self.db_handler.execute_aggregation(pipeline)
                logger.info(f"Database returned {len(results)} results")

            # Step 3: If no local data or insufficient data, fetch real ARGO data
            if len(results) < 5:  # Threshold for sufficient data
                logger.info("Fetching real-time ARGO data from global repositories")

                try:
                    # Extract region and parameters from query analysis
                    region = query_analysis.get('region', 'global')
                    parameters = query_analysis.get('parameters', ['temperature'])
                    parameter = parameters[0] if parameters else 'temperature'

                    # Fetch real ARGO data
                    real_data = self.real_argo_fetcher.fetch_real_data_for_region(
                        region=region,
                        parameter=parameter,
                        max_profiles=50
                    )

                    if real_data:
                        # Insert new data into database for future queries
                        self.db_handler.insert_argo_data(real_data)

                        # Combine with existing results
                        results.extend(real_data)
                        data_source = "real_argo_api"
                        logger.info(f"Added {len(real_data)} real ARGO profiles")

                except Exception as e:
                    logger.warning(f"Real ARGO data fetch failed: {e}")

            # Step 4: Final fallback - ensure we always have data
            if len(results) == 0:
                logger.info("Generating realistic oceanographic data as fallback")
                region = query_analysis.get('region', 'Arabian Sea')
                parameter = query_analysis.get('parameters', ['temperature'])[0] if query_analysis.get('parameters') else 'temperature'
                results = self.real_argo_fetcher._generate_realistic_data(region, parameter, 30)
                data_source = "realistic_fallback"

            # Prepare comprehensive response
            response = {
                "success": True,
                "data": results,
                "pipeline": pipeline if pipeline else [],
                "generation_method": generation_method if pipeline else "global_fetch",
                "query": query,
                "context_used": bool(context),
                "total_results": len(results),
                "data_source": data_source,
                "query_analysis": query_analysis
            }

            logger.info(f"Query processed successfully: {len(results)} results from {data_source}")
            return response

        except Exception as e:
            logger.error(f"Error processing natural query: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "query": query
            }

    def get_query_suggestions(self, partial_query: str = "") -> List[str]:
        """Get intelligent query suggestions"""
        base_suggestions = [
            "Show me temperature profiles near Mumbai",
            "Find all floats in Arabian Sea from last 3 months",
            "Compare salinity between Arabian Sea and Bay of Bengal",
            "What's the average temperature at 100m depth?",
            "Show BGC parameters in Bay of Bengal",
            "Display oxygen levels near the equator",
            "Find chlorophyll data in Arabian Sea",
            "Show floats with recent data in Indian Ocean",
            "Compare temperature trends between regions",
            "What are the nearest floats to Chennai?"
        ]

        if partial_query:
            # Filter suggestions based on partial query
            filtered = [s for s in base_suggestions if partial_query.lower() in s.lower()]
            return filtered[:5] if filtered else base_suggestions[:5]

        return base_suggestions[:5]

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and extract key components"""
        query_lower = query.lower()

        intent_analysis = {
            'query_type': 'general',
            'spatial_component': False,
            'temporal_component': False,
            'parameter_component': False,
            'comparison_component': False,
            'export_component': False,
            'statistical_component': False
        }

        # Spatial analysis
        if any(city in query_lower for city in config.CITIES.keys()):
            intent_analysis['spatial_component'] = True
            intent_analysis['query_type'] = 'spatial'

        # Temporal analysis
        if any(term in query_lower for term in ['last', 'month', 'year', 'recent', 'march', 'january', 'time']):
            intent_analysis['temporal_component'] = True
            if intent_analysis['query_type'] == 'general':
                intent_analysis['query_type'] = 'temporal'

        # Parameter analysis
        if any(param in query_lower for param in ['temperature', 'salinity', 'oxygen', 'chlorophyll', 'bgc']):
            intent_analysis['parameter_component'] = True
            if intent_analysis['query_type'] == 'general':
                intent_analysis['query_type'] = 'parameter'

        # Comparison analysis
        if any(term in query_lower for term in ['compare', 'between', 'versus', 'vs', 'difference']):
            intent_analysis['comparison_component'] = True
            intent_analysis['query_type'] = 'comparison'

        # Export analysis
        if any(term in query_lower for term in ['export', 'download', 'save', 'ascii', 'netcdf']):
            intent_analysis['export_component'] = True
            intent_analysis['query_type'] = 'export'

        # Statistical analysis
        if any(term in query_lower for term in ['average', 'mean', 'statistics', 'count', 'minimum', 'maximum']):
            intent_analysis['statistical_component'] = True
            if intent_analysis['query_type'] == 'general':
                intent_analysis['query_type'] = 'statistical'

        return intent_analysis

    def enhance_query_with_context(self, query: str, intent_analysis: Dict[str, Any]) -> str:
        """Enhance query with relevant context based on intent"""
        enhanced_query = query

        # Add spatial context if needed
        if intent_analysis['spatial_component']:
            location_info = self.extract_location_from_query(query)
            if location_info and 'city' in location_info:
                city = location_info['city']
                enhanced_query += f" (Location: {city} at {location_info['latitude']:.2f}°N, {location_info['longitude']:.2f}°E)"

        # Add temporal context if needed
        if intent_analysis['temporal_component']:
            temporal_info = self.extract_temporal_from_query(query)
            if temporal_info:
                enhanced_query += f" (Time range: {temporal_info['start_date'].strftime('%Y-%m-%d')} to {temporal_info['end_date'].strftime('%Y-%m-%d')})"

        return enhanced_query

    def validate_pipeline(self, pipeline: List[Dict]) -> Tuple[bool, str]:
        """Validate MongoDB aggregation pipeline"""
        try:
            if not pipeline:
                return False, "Empty pipeline"

            # Check for basic structure
            for stage in pipeline:
                if not isinstance(stage, dict):
                    return False, "Pipeline stage must be a dictionary"

                # Check if stage has valid operator
                stage_keys = list(stage.keys())
                if not stage_keys or not stage_keys[0].startswith('$'):
                    return False, f"Invalid stage operator: {stage_keys[0] if stage_keys else 'None'}"

            return True, "Valid pipeline"

        except Exception as e:
            return False, f"Pipeline validation error: {e}"

    def process_export_request(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        """Process data export requests"""
        from core.netcdf_processor import get_netcdf_processor

        processor = get_netcdf_processor()
        export_info = {"exported": False, "format": None, "file_path": None}

        query_lower = query.lower()

        if "ascii" in query_lower:
            output_file = Path("exports") / f"argo_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_file.parent.mkdir(exist_ok=True)

            if processor.export_to_ascii(results, output_file):
                export_info.update({
                    "exported": True,
                    "format": "ASCII",
                    "file_path": str(output_file),
                    "records": len(results)
                })

        elif "netcdf" in query_lower:
            output_file = Path("exports") / f"argo_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"
            output_file.parent.mkdir(exist_ok=True)

            if processor.export_to_netcdf(results, output_file):
                export_info.update({
                    "exported": True,
                    "format": "NetCDF",
                    "file_path": str(output_file),
                    "records": len(results)
                })

        return export_info

    def analyze_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to analyze and extract parameters from natural language query"""
        try:
            analysis_prompt = f"""
            Analyze this oceanographic query and extract key parameters:

            Query: "{query}"

            Extract and return JSON with these fields:
            - region: specific ocean region mentioned (e.g., "Arabian Sea", "Pacific", "global")
            - parameters: list of oceanographic parameters (temperature, salinity, oxygen, etc.)
            - location: any specific location mentioned (coordinates, city names)
            - temporal: time period mentioned (recent, last month, specific dates)
            - depth_range: any depth information mentioned
            - analysis_type: comparison, trends, statistics, export, etc.
            - global_scope: true if query implies global coverage

            Return only valid JSON.
            """

            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": analysis_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()

            # Clean and parse JSON
            result_text = re.sub(r'```json\s*|\s*```', '', result_text)
            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback parsing
                return self._fallback_query_analysis(query)

        except Exception as e:
            logger.error(f"Error in LLM query analysis: {e}")
            return self._fallback_query_analysis(query)

    def _fallback_query_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback query analysis using simple pattern matching"""
        query_lower = query.lower()

        analysis = {
            'region': 'global',
            'parameters': [],
            'location': None,
            'temporal': None,
            'depth_range': None,
            'analysis_type': 'general',
            'global_scope': True
        }

        # Extract region
        for region in ['arabian sea', 'bay of bengal', 'pacific', 'atlantic', 'indian ocean', 'mediterranean']:
            if region in query_lower:
                analysis['region'] = region.title()
                analysis['global_scope'] = False
                break

        # Extract parameters
        params = ['temperature', 'salinity', 'oxygen', 'chlorophyll', 'nitrate', 'bgc']
        analysis['parameters'] = [p for p in params if p in query_lower]

        # Extract cities
        cities = ['mumbai', 'chennai', 'kolkata', 'new york', 'london', 'tokyo']
        for city in cities:
            if city in query_lower:
                analysis['location'] = city
                analysis['global_scope'] = False
                break

        # Extract temporal
        temporal_terms = ['recent', 'last month', 'last year', '2023', '2024']
        for term in temporal_terms:
            if term in query_lower:
                analysis['temporal'] = term
                break

        # Extract analysis type
        if any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            analysis['analysis_type'] = 'comparison'
        elif any(word in query_lower for word in ['trend', 'time series', 'temporal']):
            analysis['analysis_type'] = 'trends'
        elif any(word in query_lower for word in ['export', 'download', 'save']):
            analysis['analysis_type'] = 'export'

        return analysis

    def extract_global_query_params(self, analysis: Dict, query: str) -> Dict[str, Any]:
        """Extract parameters for global data fetching"""
        try:
            params = {
                'max_profiles': 50,
                'parameter': 'temperature'  # Default
            }

            # Extract region
            region = analysis.get('region', 'global')
            if region and region != 'global':
                params['region'] = region

            # Extract parameters
            parameters = analysis.get('parameters', [])
            if parameters:
                params['parameter'] = parameters[0]  # Use first parameter

            # Extract location from query
            location = self.extract_location_from_query(query)
            if location:
                lat, lon = location['latitude'], location['longitude']
                # Create bounding box around location
                params['lat_range'] = (lat - 5, lat + 5)
                params['lon_range'] = (lon - 5, lon + 5)

            # Extract temporal
            temporal = self.extract_temporal_from_query(query)
            if temporal:
                params['date_range'] = (temporal['start_date'], temporal['end_date'])

            # Global scope handling
            if analysis.get('global_scope', True):
                params['max_profiles'] = 100  # More data for global queries

            return params

        except Exception as e:
            logger.error(f"Error extracting global params: {e}")
            return {'region': 'global', 'parameter': 'temperature', 'max_profiles': 50}

def get_ai_engine():
    """Factory function to get AI engine instance"""
    return AdvancedAIEngine()