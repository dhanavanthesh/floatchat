"""
ARGO NetCDF Data Processor
Real-time ingestion and processing of ARGO float data from NetCDF files
"""

import ftplib
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import xarray as xr
import pandas as pd
from config import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class ARGONetCDFProcessor:
    """Process ARGO NetCDF files and convert to MongoDB-ready format"""

    def __init__(self):
        self.ftp_host = config.ARGO_FTP_HOST
        self.ftp_path = config.ARGO_FTP_PATH
        self.netcdf_dir = config.NETCDF_DIR
        self.processed_dir = config.PROCESSED_DIR

    async def download_argo_index(self) -> List[str]:
        """Download ARGO index files to get available data"""
        try:
            logger.info("Connecting to ARGO FTP server...")
            ftp = ftplib.FTP(self.ftp_host, timeout=30)
            ftp.login()  # Anonymous login

            # Try multiple index paths
            index_paths = [
                "ar_index_global_prof.txt",
                "argo_index_global_prof.txt",
                "dac/ar_index_global_prof.txt"
            ]

            index_file = None
            for index_path in index_paths:
                try:
                    # Download index file
                    index_file = self.netcdf_dir / "ar_index_global_prof.txt"
                    with open(index_file, 'wb') as f:
                        ftp.retrbinary(f'RETR {index_path}', f.write)
                    logger.info(f"Downloaded ARGO index from {index_path}")
                    break
                except ftplib.error_perm as e:
                    logger.warning(f"Failed to download from {index_path}: {e}")
                    continue

            ftp.quit()

            if not index_file or not index_file.exists():
                logger.warning("Could not download ARGO index, generating sample data")
                return self.generate_sample_file_list()

            # Parse index file
            return self.parse_argo_index(index_file)

        except Exception as e:
            logger.error(f"Error downloading ARGO index: {e}")
            return self.generate_sample_file_list()

    def parse_argo_index(self, index_file: Path) -> List[str]:
        """Parse ARGO index file to get relevant NetCDF files"""
        try:
            # Read index file
            df = pd.read_csv(index_file, sep=',', skiprows=8)

            # Filter for Indian Ocean region
            indian_ocean_files = []
            for _, row in df.iterrows():
                lat, lon = row['latitude'], row['longitude']

                # Check if in our regions of interest
                for region_name, region_info in config.REGIONS.items():
                    lat_bounds = region_info['bounds']['lat']
                    lon_bounds = region_info['bounds']['lon']

                    if (lat_bounds[0] <= lat <= lat_bounds[1] and
                        lon_bounds[0] <= lon <= lon_bounds[1]):
                        indian_ocean_files.append(row['file'])
                        break

            logger.info(f"Found {len(indian_ocean_files)} relevant ARGO files")
            return indian_ocean_files[:100]  # Limit for demo

        except Exception as e:
            logger.error(f"Error parsing index file: {e}")
            return []

    async def download_netcdf_file(self, file_path: str) -> Optional[Path]:
        """Download a single NetCDF file"""
        try:
            local_file = self.netcdf_dir / Path(file_path).name

            # Skip if already downloaded
            if local_file.exists():
                return local_file

            ftp = ftplib.FTP(self.ftp_host)
            ftp.login()

            with open(local_file, 'wb') as f:
                ftp.retrbinary(f'RETR {self.ftp_path}/{file_path}', f.write)

            ftp.quit()
            logger.info(f"Downloaded {local_file}")
            return local_file

        except Exception as e:
            logger.error(f"Error downloading {file_path}: {e}")
            return None

    def process_netcdf_file(self, netcdf_file: Path) -> List[Dict[str, Any]]:
        """Process a single NetCDF file into MongoDB documents"""
        try:
            # Open NetCDF file with xarray
            ds = xr.open_dataset(netcdf_file)

            # Extract metadata
            float_id = str(ds.attrs.get('platform_number', ''))
            platform_type = str(ds.attrs.get('platform_type', 'UNKNOWN'))
            institution = str(ds.attrs.get('institution', 'UNKNOWN'))
            project = str(ds.attrs.get('project_name', 'ARGO'))

            documents = []

            # Process each profile (cycle)
            for cycle_idx, cycle in enumerate(ds.CYCLE_NUMBER.values):
                try:
                    # Extract position and time
                    lat = float(ds.LATITUDE.values[cycle_idx])
                    lon = float(ds.LONGITUDE.values[cycle_idx])

                    # Handle time - convert to datetime
                    julian_day = float(ds.JULD.values[cycle_idx])
                    if not np.isnan(julian_day):
                        # Convert Julian day to datetime (ARGO reference: 1950-01-01)
                        reference_date = datetime(1950, 1, 1)
                        timestamp = reference_date + timedelta(days=julian_day)
                    else:
                        timestamp = datetime.now()

                    # Determine region
                    region = self.get_region(lat, lon)
                    if not region:
                        continue  # Skip if not in our regions of interest

                    # Extract measurements
                    measurements = self.extract_measurements(ds, cycle_idx)
                    if not measurements:
                        continue

                    # Create document
                    document = {
                        '_id': f"float_{float_id}_{cycle}_{timestamp.strftime('%Y%m%d')}",
                        'float_id': float_id,
                        'cycle_number': int(cycle),
                        'location': {
                            'type': 'Point',
                            'coordinates': [lon, lat]
                        },
                        'timestamp': timestamp,
                        'region': region,
                        'platform_type': platform_type,
                        'data_mode': str(ds.DATA_MODE.values[cycle_idx] if 'DATA_MODE' in ds else 'R'),
                        'measurements': measurements,
                        'quality_flags': self.extract_quality_flags(ds, cycle_idx),
                        'metadata': {
                            'institution': institution,
                            'project': project,
                            'netcdf_file': str(netcdf_file.name),
                            'processing_date': datetime.now(),
                            'source': 'real_argo_netcdf'
                        }
                    }

                    documents.append(document)

                except Exception as e:
                    logger.warning(f"Error processing cycle {cycle_idx}: {e}")
                    continue

            ds.close()
            logger.info(f"Processed {len(documents)} profiles from {netcdf_file.name}")
            return documents

        except Exception as e:
            logger.error(f"Error processing NetCDF file {netcdf_file}: {e}")
            return []

    def extract_measurements(self, ds: xr.Dataset, cycle_idx: int) -> List[Dict[str, Any]]:
        """Extract measurement data for a specific cycle"""
        measurements = []

        try:
            # Get all depth levels for this cycle
            pres_values = ds.PRES.values[cycle_idx, :]
            temp_values = ds.TEMP.values[cycle_idx, :]
            sal_values = ds.PSAL.values[cycle_idx, :]

            # BGC parameters (if available)
            oxy_values = ds.DOXY.values[cycle_idx, :] if 'DOXY' in ds else None
            chla_values = ds.CHLA.values[cycle_idx, :] if 'CHLA' in ds else None
            nitrate_values = ds.NITRATE.values[cycle_idx, :] if 'NITRATE' in ds else None

            for level_idx, pressure in enumerate(pres_values):
                if np.isnan(pressure) or pressure < 0:
                    continue

                # Extract values for this depth level
                temp = temp_values[level_idx]
                sal = sal_values[level_idx]

                # Skip if essential parameters are missing
                if np.isnan(temp) and np.isnan(sal):
                    continue

                measurement = {
                    'depth': float(pressure),  # Using pressure as depth approximation
                    'pressure': float(pressure) if not np.isnan(pressure) else None,
                    'temperature': float(temp) if not np.isnan(temp) else None,
                    'salinity': float(sal) if not np.isnan(sal) else None
                }

                # Add BGC parameters if available
                if oxy_values is not None and not np.isnan(oxy_values[level_idx]):
                    measurement['oxygen'] = float(oxy_values[level_idx])

                if chla_values is not None and not np.isnan(chla_values[level_idx]):
                    measurement['chlorophyll'] = float(chla_values[level_idx])

                if nitrate_values is not None and not np.isnan(nitrate_values[level_idx]):
                    measurement['nitrate'] = float(nitrate_values[level_idx])

                measurements.append(measurement)

            return measurements

        except Exception as e:
            logger.error(f"Error extracting measurements: {e}")
            return []

    def extract_quality_flags(self, ds: xr.Dataset, cycle_idx: int) -> Dict[str, str]:
        """Extract quality flags for measurements"""
        quality_flags = {}

        try:
            # Temperature quality
            if 'TEMP_QC' in ds:
                temp_qc = ds.TEMP_QC.values[cycle_idx, 0] if ds.TEMP_QC.values.size > 0 else '1'
                quality_flags['temperature'] = config.QUALITY_FLAGS.get(int(temp_qc), 'unknown')

            # Salinity quality
            if 'PSAL_QC' in ds:
                sal_qc = ds.PSAL_QC.values[cycle_idx, 0] if ds.PSAL_QC.values.size > 0 else '1'
                quality_flags['salinity'] = config.QUALITY_FLAGS.get(int(sal_qc), 'unknown')

            # Pressure quality
            if 'PRES_QC' in ds:
                pres_qc = ds.PRES_QC.values[cycle_idx, 0] if ds.PRES_QC.values.size > 0 else '1'
                quality_flags['pressure'] = config.QUALITY_FLAGS.get(int(pres_qc), 'unknown')

            return quality_flags

        except Exception as e:
            logger.warning(f"Error extracting quality flags: {e}")
            return {'temperature': 'unknown', 'salinity': 'unknown', 'pressure': 'unknown'}

    def get_region(self, lat: float, lon: float) -> Optional[str]:
        """Determine which region a coordinate belongs to"""
        for region_name, region_info in config.REGIONS.items():
            lat_bounds = region_info['bounds']['lat']
            lon_bounds = region_info['bounds']['lon']

            if (lat_bounds[0] <= lat <= lat_bounds[1] and
                lon_bounds[0] <= lon <= lon_bounds[1]):
                return region_name

        return None

    async def process_multiple_files(self, file_list: List[str], max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Process multiple NetCDF files concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        all_documents = []

        async def process_single_file(file_path: str):
            async with semaphore:
                netcdf_file = await self.download_netcdf_file(file_path)
                if netcdf_file:
                    return self.process_netcdf_file(netcdf_file)
                return []

        # Process files concurrently
        tasks = [process_single_file(file_path) for file_path in file_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all documents
        for result in results:
            if isinstance(result, list):
                all_documents.extend(result)

        logger.info(f"Processed {len(all_documents)} total documents from {len(file_list)} files")
        return all_documents

    def generate_sample_file_list(self) -> List[str]:
        """Generate sample file paths for demo purposes"""
        return [
            "dac/incois/2901623/profiles/R2901623_001.nc",
            "dac/incois/2901623/profiles/R2901623_002.nc",
            "dac/incois/2901624/profiles/R2901624_001.nc",
            "dac/incois/2901625/profiles/R2901625_001.nc",
            "dac/incois/2901626/profiles/R2901626_001.nc"
        ]

    def export_to_ascii(self, documents: List[Dict[str, Any]], output_file: Path) -> bool:
        """Export processed data to ASCII format"""
        try:
            with open(output_file, 'w') as f:
                # Write header
                f.write("# ARGO Float Data Export\n")
                f.write("# Generated on: {}\n".format(datetime.now().isoformat()))
                f.write("# Format: Float_ID, Cycle, Timestamp, Latitude, Longitude, Region, Depth, Temperature, Salinity, Oxygen, Chlorophyll, Nitrate\n")
                f.write("Float_ID,Cycle,Timestamp,Latitude,Longitude,Region,Depth,Temperature,Salinity,Oxygen,Chlorophyll,Nitrate\n")

                for doc in documents:
                    float_id = doc['float_id']
                    cycle = doc['cycle_number']
                    timestamp = doc['timestamp'].isoformat()
                    lat = doc['location']['coordinates'][1]
                    lon = doc['location']['coordinates'][0]
                    region = doc['region']

                    for measurement in doc['measurements']:
                        depth = measurement.get('depth', '')
                        temp = measurement.get('temperature', '')
                        sal = measurement.get('salinity', '')
                        oxy = measurement.get('oxygen', '')
                        chla = measurement.get('chlorophyll', '')
                        nitrate = measurement.get('nitrate', '')

                        f.write(f"{float_id},{cycle},{timestamp},{lat},{lon},{region},{depth},{temp},{sal},{oxy},{chla},{nitrate}\n")

            logger.info(f"Exported {len(documents)} documents to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to ASCII: {e}")
            return False

    def export_to_netcdf(self, documents: List[Dict[str, Any]], output_file: Path) -> bool:
        """Export processed data back to NetCDF format"""
        try:
            # Prepare data arrays
            all_data = []
            for doc in documents:
                for measurement in doc['measurements']:
                    all_data.append({
                        'float_id': doc['float_id'],
                        'cycle_number': doc['cycle_number'],
                        'timestamp': doc['timestamp'],
                        'latitude': doc['location']['coordinates'][1],
                        'longitude': doc['location']['coordinates'][0],
                        'region': doc['region'],
                        'depth': measurement.get('depth'),
                        'temperature': measurement.get('temperature'),
                        'salinity': measurement.get('salinity'),
                        'oxygen': measurement.get('oxygen'),
                        'chlorophyll': measurement.get('chlorophyll'),
                        'nitrate': measurement.get('nitrate')
                    })

            # Convert to DataFrame and then to xarray Dataset
            df = pd.DataFrame(all_data)

            # Create xarray Dataset
            ds = xr.Dataset({
                'temperature': (['obs'], df['temperature'].values),
                'salinity': (['obs'], df['salinity'].values),
                'pressure': (['obs'], df['depth'].values),
                'oxygen': (['obs'], df['oxygen'].values),
                'chlorophyll': (['obs'], df['chlorophyll'].values),
                'nitrate': (['obs'], df['nitrate'].values),
                'latitude': (['obs'], df['latitude'].values),
                'longitude': (['obs'], df['longitude'].values),
                'float_id': (['obs'], df['float_id'].values),
                'cycle_number': (['obs'], df['cycle_number'].values)
            }, coords={
                'obs': range(len(df))
            })

            # Add attributes
            ds.attrs['title'] = 'ARGO Float Data Export'
            ds.attrs['institution'] = 'FloatChat System'
            ds.attrs['source'] = 'Processed ARGO data'
            ds.attrs['history'] = f'Created on {datetime.now().isoformat()}'

            # Save to NetCDF
            ds.to_netcdf(output_file)
            logger.info(f"Exported {len(documents)} documents to NetCDF: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to NetCDF: {e}")
            return False

def get_netcdf_processor():
    """Factory function to get NetCDF processor instance"""
    return ARGONetCDFProcessor()