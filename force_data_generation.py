"""
Force generate comprehensive ARGO data for testing
"""

from core.database import get_database_handler
from core.argo_realtime import get_real_argo_fetcher
from core.met_data import get_met_context
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def force_generate_data():
    """Force generate comprehensive test data"""
    try:
        # Initialize components
        db_handler = get_database_handler()
        fetcher = get_real_argo_fetcher()
        met_context = get_met_context()

        print("Generating comprehensive ARGO data...")

        # Generate data for multiple regions
        all_data = []

        # Arabian Sea data
        print("Generating Arabian Sea data...")
        arabian_data = fetcher._generate_realistic_data("Arabian Sea", "temperature", 30)
        all_data.extend(arabian_data)

        # Bay of Bengal data
        print("Generating Bay of Bengal data...")
        bengal_data = fetcher._generate_realistic_data("Bay of Bengal", "temperature", 30)
        all_data.extend(bengal_data)

        # Add other regions
        print("Generating global ocean data...")
        pacific_data = fetcher._generate_realistic_data("Pacific Ocean", "temperature", 20)
        all_data.extend(pacific_data)

        atlantic_data = fetcher._generate_realistic_data("Atlantic Ocean", "temperature", 20)
        all_data.extend(atlantic_data)

        # Add meteorological context
        print("Adding meteorological context...")
        all_data = met_context.add_met_context_to_data(all_data)

        # Insert into database
        print("Inserting into database...")
        success = db_handler.insert_argo_data(all_data)

        if success:
            print(f"✅ Successfully generated {len(all_data)} comprehensive profiles!")

            # Verify data
            stats = db_handler.get_database_statistics()
            print(f"Database now contains {stats.get('total_profiles', 0)} total profiles")

            # Check data structure
            sample = all_data[0]
            print(f"Sample profile has {len(sample['measurements'])} measurements")
            print(f"Sample measurement: {sample['measurements'][0]}")

            # Count BGC profiles
            bgc_count = sum(1 for profile in all_data
                           if any('oxygen' in m for m in profile['measurements']))
            print(f"BGC profiles: {bgc_count}/{len(all_data)}")

            return True
        else:
            print("❌ Database insertion failed")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    force_generate_data()