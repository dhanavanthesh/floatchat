"""
Unified Database Handler
MongoDB + ChromaDB integration for ARGO data storage and retrieval
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pymongo
from pymongo import MongoClient, GEOSPHERE, TEXT
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import numpy as np
from config import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class UnifiedDatabaseHandler:
    """Unified handler for MongoDB (main data) + ChromaDB (vector search)"""

    def __init__(self):
        self.mongodb_uri = config.MONGODB_URI
        self.database_name = config.MONGODB_DATABASE
        self.collection_name = config.MONGODB_COLLECTION
        self.chromadb_path = config.CHROMADB_PATH

        # Initialize connections
        self.mongo_client = None
        self.mongo_db = None
        self.mongo_collection = None
        self.chroma_client = None
        self.chroma_collection = None

        self.connect()

    def connect(self):
        """Establish connections to both databases"""
        try:
            # MongoDB connection
            self.mongo_client = MongoClient(
                self.mongodb_uri,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50
            )
            self.mongo_client.server_info()  # Test connection
            self.mongo_db = self.mongo_client[self.database_name]
            self.mongo_collection = self.mongo_db[self.collection_name]

            # Setup MongoDB indexes
            self.setup_mongodb_indexes()
            logger.info("✅ MongoDB connected successfully")

            # ChromaDB connection
            self.setup_chromadb()
            logger.info("✅ ChromaDB connected successfully")

        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    def setup_mongodb_indexes(self):
        """Create optimized indexes for ARGO data queries"""
        try:
            # Geospatial index for location queries
            self.mongo_collection.create_index([("location", GEOSPHERE)])

            # Temporal index
            self.mongo_collection.create_index([("timestamp", pymongo.ASCENDING)])

            # Float identification
            self.mongo_collection.create_index([("float_id", pymongo.ASCENDING)])

            # Regional queries
            self.mongo_collection.create_index([("region", pymongo.ASCENDING)])

            # Compound indexes for common queries
            self.mongo_collection.create_index([
                ("region", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ])

            self.mongo_collection.create_index([
                ("float_id", pymongo.ASCENDING),
                ("cycle_number", pymongo.ASCENDING)
            ], unique=True)

            # Text search index for metadata
            self.mongo_collection.create_index([
                ("metadata.institution", TEXT),
                ("metadata.project", TEXT),
                ("platform_type", TEXT)
            ])

            logger.info("MongoDB indexes created successfully")

        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")

    def setup_chromadb(self):
        """Setup ChromaDB for vector search and RAG"""
        try:
            # Skip ChromaDB if not available
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError:
                logger.warning("ChromaDB not available, skipping vector search")
                self.chroma_collection = None
                return

            self.chroma_client = chromadb.PersistentClient(
                path=self.chromadb_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection
            try:
                existing_collections = [col.name for col in self.chroma_client.list_collections()]
                if config.CHROMADB_COLLECTION in existing_collections:
                    self.chroma_collection = self.chroma_client.get_collection(config.CHROMADB_COLLECTION)
                    logger.info("Found existing ChromaDB collection")
                else:
                    raise ValueError("Collection does not exist")
            except (ValueError, Exception):
                self.chroma_collection = self.chroma_client.create_collection(
                    name=config.CHROMADB_COLLECTION,
                    metadata={"description": "ARGO oceanographic context and knowledge base"}
                )
                self.populate_chroma_context()
                logger.info("ChromaDB collection created and populated")

        except Exception as e:
            logger.error(f"ChromaDB setup error: {e}")
            self.chroma_collection = None

    def populate_chroma_context(self):
        """Populate ChromaDB with oceanographic knowledge"""
        context_data = [
            # Geographic locations
            {
                "id": "mumbai_location",
                "text": "Mumbai (19.0760°N, 72.8777°E) is a major coastal city in western India, located on the Arabian Sea. The waters near Mumbai show typical Arabian Sea characteristics with high salinity (35.5-36.5 PSU) and warm surface temperatures (26-30°C).",
                "metadata": {"type": "location", "region": "Arabian Sea", "city": "mumbai"}
            },
            {
                "id": "chennai_location",
                "text": "Chennai (13.0827°N, 80.2707°E) is located on India's southeastern coast on the Bay of Bengal. Bay of Bengal waters are characterized by lower salinity (33-35 PSU) due to freshwater input from rivers and monsoon precipitation.",
                "metadata": {"type": "location", "region": "Bay of Bengal", "city": "chennai"}
            },

            # Oceanographic processes
            {
                "id": "arabian_sea_characteristics",
                "text": "Arabian Sea: High salinity waters (35.5-36.5 PSU), strong seasonal variability due to monsoons, upwelling along western Indian coast, oxygen minimum zone at intermediate depths (200-1000m), typical surface temperatures 24-30°C.",
                "metadata": {"type": "region", "region": "Arabian Sea"}
            },
            {
                "id": "bay_of_bengal_characteristics",
                "text": "Bay of Bengal: Lower salinity (33-35 PSU) due to river discharge, strong stratification, cyclone activity, warmer surface waters (26-31°C), less mixing than Arabian Sea, significant freshwater input from Ganges-Brahmaputra.",
                "metadata": {"type": "region", "region": "Bay of Bengal"}
            },

            # Depth zones
            {
                "id": "surface_layer",
                "text": "Surface layer (0-50m): Mixed layer, high temperature variability, direct interaction with atmosphere, seasonal thermocline formation, primary productivity zone.",
                "metadata": {"type": "depth", "zone": "surface", "range": "0-50"}
            },
            {
                "id": "thermocline",
                "text": "Thermocline (50-200m): Rapid temperature gradient, density stratification, barrier to vertical mixing, important for marine ecosystems, seasonal variation in depth.",
                "metadata": {"type": "depth", "zone": "thermocline", "range": "50-200"}
            },
            {
                "id": "intermediate_waters",
                "text": "Intermediate waters (200-1000m): More stable temperatures (5-15°C), oxygen minimum zones, important water mass boundaries, less seasonal variability.",
                "metadata": {"type": "depth", "zone": "intermediate", "range": "200-1000"}
            },

            # BGC parameters
            {
                "id": "oxygen_dynamics",
                "text": "Dissolved oxygen: Critical for marine life, shows strong vertical gradients, oxygen minimum zones in Arabian Sea and Bay of Bengal at 200-1000m depth, affected by biological processes and circulation.",
                "metadata": {"type": "parameter", "param": "oxygen"}
            },
            {
                "id": "chlorophyll_patterns",
                "text": "Chlorophyll-a: Indicator of phytoplankton biomass, highest in surface waters, shows seasonal patterns linked to monsoons, upwelling events, and nutrient availability.",
                "metadata": {"type": "parameter", "param": "chlorophyll"}
            },

            # Seasonal patterns
            {
                "id": "monsoon_effects",
                "text": "Monsoon impacts: Southwest monsoon (June-September) brings intense mixing, upwelling, and productivity. Northeast monsoon (October-February) affects Bay of Bengal more. Seasonal reversals in surface currents.",
                "metadata": {"type": "seasonal", "season": "monsoon"}
            },

            # ARGO float technology
            {
                "id": "argo_float_basics",
                "text": "ARGO floats: Autonomous profiling floats that drift with ocean currents, diving to 2000m depth every 10 days, measuring temperature, salinity, and pressure. BGC floats also measure oxygen, chlorophyll, nitrate, and pH.",
                "metadata": {"type": "technology", "system": "argo"}
            }
        ]

        # Extract data for ChromaDB
        texts = [item["text"] for item in context_data]
        metadatas = [item["metadata"] for item in context_data]
        ids = [item["id"] for item in context_data]

        # Add to collection
        self.chroma_collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(context_data)} context documents to ChromaDB")

    # MongoDB Operations
    def insert_argo_data(self, documents: List[Dict[str, Any]]) -> bool:
        """Insert ARGO data documents into MongoDB"""
        try:
            if not documents:
                return False

            # Prepare documents for insertion
            prepared_docs = []
            for doc in documents:
                # Ensure timestamp is datetime object
                if isinstance(doc.get('timestamp'), str):
                    doc['timestamp'] = datetime.fromisoformat(doc['timestamp'].replace('Z', '+00:00'))
                prepared_docs.append(doc)

            # Insert with upsert to handle duplicates
            bulk_operations = []
            for doc in prepared_docs:
                bulk_operations.append(
                    pymongo.UpdateOne(
                        {"_id": doc["_id"]},
                        {"$set": doc},
                        upsert=True
                    )
                )

            if bulk_operations:
                result = self.mongo_collection.bulk_write(bulk_operations, ordered=False)

                # Handle bulk write errors gracefully
                if hasattr(result, 'bulk_api_result') and result.bulk_api_result.get('writeErrors'):
                    duplicate_errors = sum(1 for error in result.bulk_api_result['writeErrors']
                                         if error.get('code') == 11000)
                    if duplicate_errors > 0:
                        logger.info(f"Skipped {duplicate_errors} duplicate records (already exist)")

                inserted_count = result.upserted_count + result.modified_count
                logger.info(f"Successfully processed {inserted_count} documents")

                if result.upserted_count > 0:
                    logger.info(f"New records inserted: {result.upserted_count}")
                if result.modified_count > 0:
                    logger.info(f"Existing records updated: {result.modified_count}")

            return True

        except Exception as e:
            # Check if it's just duplicate key errors
            if "duplicate key error" in str(e).lower() or "11000" in str(e):
                logger.info("Some duplicate records were skipped - this is normal for real-time data updates")
                return True
            else:
                logger.error(f"Error inserting ARGO data: {e}")
                return False

    def get_floats_by_region(self, region: str, limit: int = 1000) -> List[Dict]:
        """Get floats by geographic region"""
        try:
            if region.lower() == "all":
                query = {}
            else:
                query = {"region": {"$regex": region, "$options": "i"}}

            return list(self.mongo_collection.find(query).limit(limit))

        except Exception as e:
            logger.error(f"Error querying by region: {e}")
            return []

    def get_floats_near_location(self, longitude: float, latitude: float, radius_km: float = 100) -> List[Dict]:
        """Get floats near a specific location using geospatial query"""
        try:
            query = {
                "location": {
                    "$near": {
                        "$geometry": {
                            "type": "Point",
                            "coordinates": [longitude, latitude]
                        },
                        "$maxDistance": radius_km * 1000  # Convert to meters
                    }
                }
            }
            return list(self.mongo_collection.find(query))

        except Exception as e:
            logger.error(f"Error in geospatial query: {e}")
            return []

    def get_floats_by_date_range(self, start_date: datetime, end_date: datetime, region: Optional[str] = None) -> List[Dict]:
        """Get floats within a date range"""
        try:
            query = {
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }

            if region and region.lower() != "all":
                query["region"] = {"$regex": region, "$options": "i"}

            return list(self.mongo_collection.find(query))

        except Exception as e:
            logger.error(f"Error querying by date range: {e}")
            return []

    def get_profiles_by_depth_range(self, min_depth: float, max_depth: float, parameter: str = "temperature") -> List[Dict]:
        """Get profiles within specific depth range for a parameter"""
        try:
            pipeline = [
                {"$unwind": "$measurements"},
                {
                    "$match": {
                        "measurements.depth": {"$gte": min_depth, "$lte": max_depth},
                        f"measurements.{parameter}": {"$exists": True, "$ne": None}
                    }
                },
                {
                    "$group": {
                        "_id": "$_id",
                        "float_id": {"$first": "$float_id"},
                        "location": {"$first": "$location"},
                        "timestamp": {"$first": "$timestamp"},
                        "region": {"$first": "$region"},
                        "measurements": {"$push": "$measurements"}
                    }
                }
            ]

            return list(self.mongo_collection.aggregate(pipeline))

        except Exception as e:
            logger.error(f"Error querying by depth range: {e}")
            return []

    def execute_aggregation(self, pipeline: List[Dict]) -> List[Dict]:
        """Execute a custom MongoDB aggregation pipeline"""
        try:
            return list(self.mongo_collection.aggregate(pipeline))
        except Exception as e:
            logger.error(f"Error executing aggregation: {e}")
            return []

    def get_total_count(self) -> int:
        """Get total count of documents in the collection"""
        try:
            return self.mongo_collection.count_documents({})
        except Exception as e:
            logger.error(f"Error getting total count: {e}")
            return 0

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            stats_pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_profiles": {"$sum": 1},
                        "unique_floats": {"$addToSet": "$float_id"},
                        "regions": {"$addToSet": "$region"},
                        "date_range": {"$push": "$timestamp"},
                        "platform_types": {"$addToSet": "$platform_type"}
                    }
                },
                {
                    "$project": {
                        "total_profiles": 1,
                        "unique_floats": {"$size": "$unique_floats"},
                        "regions": 1,
                        "platform_types": 1,
                        "min_date": {"$min": "$date_range"},
                        "max_date": {"$max": "$date_range"}
                    }
                }
            ]

            result = list(self.mongo_collection.aggregate(stats_pipeline))
            return result[0] if result else {}

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    # ChromaDB Operations
    def search_context(self, query: str, n_results: int = 5) -> List[str]:
        """Search for relevant context using ChromaDB"""
        if not self.chroma_collection:
            return []

        try:
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results['documents'][0] if results['documents'] else []

        except Exception as e:
            logger.error(f"Error searching context: {e}")
            return []

    def add_context_document(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        """Add a new context document to ChromaDB"""
        if not self.chroma_collection:
            return False

        try:
            self.chroma_collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            return True

        except Exception as e:
            logger.error(f"Error adding context document: {e}")
            return False

    def close_connections(self):
        """Close all database connections"""
        try:
            if self.mongo_client:
                self.mongo_client.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

def get_database_handler():
    """Factory function to get database handler instance"""
    return UnifiedDatabaseHandler()