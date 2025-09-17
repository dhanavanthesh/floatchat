"""
Test Data Generation and Visualization
"""

from core.argo_realtime import get_real_argo_fetcher
from core.visualizations import get_visualizer
import pandas as pd

def test_data_and_viz():
    print("Testing ARGO data generation and visualization...")

    # Test data fetcher
    fetcher = get_real_argo_fetcher()
    print("ARGO fetcher initialized")

    # Generate test data
    print("Generating test data for Arabian Sea vs Bay of Bengal...")

    arabian_data = fetcher._generate_realistic_data("Arabian Sea", "temperature", 15)
    bengal_data = fetcher._generate_realistic_data("Bay of Bengal", "temperature", 15)

    all_data = arabian_data + bengal_data

    print(f"Generated {len(all_data)} profiles")
    print(f"   - Arabian Sea: {len(arabian_data)} profiles")
    print(f"   - Bay of Bengal: {len(bengal_data)} profiles")

    # Check data structure
    sample_profile = all_data[0]
    print(f"\nSample profile structure:")
    print(f"   - Float ID: {sample_profile['float_id']}")
    print(f"   - Region: {sample_profile['region']}")
    print(f"   - Measurements: {len(sample_profile['measurements'])} depth levels")

    # Check measurements
    sample_measurement = sample_profile['measurements'][0]
    print(f"   - Sample measurement: {sample_measurement}")

    # Count profiles with BGC data
    bgc_count = sum(1 for profile in all_data
                   if any('oxygen' in m for m in profile['measurements']))

    print(f"   - BGC profiles: {bgc_count}/{len(all_data)}")

    # Test visualization
    print("\nTesting visualization...")
    try:
        viz = get_visualizer()
        print("Visualizer initialized")

        # Test salinity comparison
        salinity_fig = viz.create_salinity_comparison_arabian_bay(all_data)
        print("Salinity comparison visualization created")

        # Test depth profile
        depth_fig = viz.create_depth_profile_plot(all_data)
        print("Depth profile visualization created")

        # Test T-S diagram
        ts_fig = viz.create_ts_diagram(all_data)
        print("T-S diagram created")

        # Test map
        map_fig = viz.create_float_trajectory_map(all_data)
        print("Map visualization created")

        print("\nAll tests passed! Data generation and visualization working correctly.")

        return True

    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

if __name__ == "__main__":
    test_data_and_viz()