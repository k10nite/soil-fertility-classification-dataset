"""
Mapping and Validation Pipeline
Validates combined_field_data.csv against final_merged_data_sorted.csv (source of truth)
Maps records using UUID and validates NPK nutrient data
"""

import pandas as pd
import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
SOURCE_CSV = REPO_ROOT / "data_prepreocessing" / "data" / "combined" / "final_merged_data_sorted.csv"
TARGET_CSV = REPO_ROOT / "organized_images" / "combined_field_data.csv"


def load_data():
    """Load both CSV files"""
    print(f"Loading source of truth: {SOURCE_CSV}")
    source_df = pd.read_csv(SOURCE_CSV)

    print(f"Loading target for validation: {TARGET_CSV}")
    target_df = pd.read_csv(TARGET_CSV)

    return source_df, target_df


def validate_uuid_mapping(source_df, target_df):
    """Validate UUID relationship between source and target"""
    source_uuids = set(source_df['uuid'])
    target_uuids = set(target_df['uuid'])

    print("\n" + "="*80)
    print("UUID MAPPING VALIDATION")
    print("="*80)

    print(f"\nSource (final_merged_data_sorted.csv): {len(source_uuids)} unique UUIDs")
    print(f"Target (combined_field_data.csv): {len(target_uuids)} unique UUIDs")

    # Find differences
    only_in_source = source_uuids - target_uuids
    only_in_target = target_uuids - source_uuids
    common_uuids = source_uuids & target_uuids

    print(f"\nCommon UUIDs: {len(common_uuids)}")
    print(f"Only in source: {len(only_in_source)}")
    print(f"Only in target: {len(only_in_target)}")

    if only_in_source:
        print(f"\nFirst 10 UUIDs only in source:")
        for uuid in list(only_in_source)[:10]:
            print(f"  - {uuid}")

    if only_in_target:
        print(f"\nFirst 10 UUIDs only in target:")
        for uuid in list(only_in_target)[:10]:
            print(f"  - {uuid}")

    return common_uuids


def compare_common_columns(source_df, target_df, common_uuids):
    """Compare common columns for matching UUIDs"""
    print("\n" + "="*80)
    print("COMMON COLUMNS VALIDATION")
    print("="*80)

    # Get common columns (excluding NPK columns)
    source_cols = set(source_df.columns)
    target_cols = set(target_df.columns)
    npk_cols = {'ph', 'k', 'p', 'n'}

    common_cols = (source_cols & target_cols) - npk_cols
    print(f"\nCommon columns to validate: {len(common_cols)}")
    print(f"Columns: {sorted(common_cols)}")

    # Create indexed dataframes for comparison
    source_indexed = source_df.set_index('uuid')
    target_indexed = target_df.set_index('uuid')

    # Compare each common column
    mismatches = {}
    for col in common_cols:
        if col == 'uuid':
            continue

        # Get values for common UUIDs
        source_vals = source_indexed.loc[list(common_uuids), col]
        target_vals = target_indexed.loc[list(common_uuids), col]

        # Convert to string for comparison (handles float/int differences)
        source_str = source_vals.astype(str)
        target_str = target_vals.astype(str)

        # Find mismatches
        col_mismatches = source_str != target_str
        mismatch_count = col_mismatches.sum()

        if mismatch_count > 0:
            mismatches[col] = mismatch_count
            print(f"\n[WARNING] {col}: {mismatch_count} mismatches")

            # Show first few examples
            mismatch_uuids = source_str[col_mismatches].index[:3]
            for uuid in mismatch_uuids:
                print(f"    UUID {uuid}:")
                print(f"      Source: {source_indexed.loc[uuid, col]}")
                print(f"      Target: {target_indexed.loc[uuid, col]}")

    if not mismatches:
        print("\n[OK] All common columns match perfectly!")
    else:
        print(f"\n[WARNING] Total columns with mismatches: {len(mismatches)}")

    return mismatches


def analyze_npk_data(source_df):
    """Analyze NPK nutrient data in source"""
    print("\n" + "="*80)
    print("NPK NUTRIENT DATA ANALYSIS")
    print("="*80)

    npk_cols = ['ph', 'k', 'p', 'n']

    print(f"\nTotal records in source: {len(source_df)}")

    for col in npk_cols:
        if col in source_df.columns:
            non_null = source_df[col].notna().sum()
            null = source_df[col].isna().sum()
            unique_vals = source_df[col].nunique()

            print(f"\n{col.upper()}:")
            print(f"  Non-null values: {non_null} ({non_null/len(source_df)*100:.1f}%)")
            print(f"  Null values: {null} ({null/len(source_df)*100:.1f}%)")
            print(f"  Unique values: {unique_vals}")

            if non_null > 0:
                print(f"  Min: {source_df[col].min()}")
                print(f"  Max: {source_df[col].max()}")
                print(f"  Mean: {source_df[col].mean():.2f}")
                print(f"  Value distribution:")
                value_counts = source_df[col].value_counts().head(10)
                for val, count in value_counts.items():
                    print(f"    {val}: {count} records")


def generate_mapping_report(source_df, target_df, common_uuids):
    """Generate comprehensive mapping report"""
    print("\n" + "="*80)
    print("MAPPING SUMMARY REPORT")
    print("="*80)

    # Create mapping dataframe
    source_indexed = source_df.set_index('uuid')

    # Get NPK data for common UUIDs
    common_list = list(common_uuids)
    mapping_df = source_indexed.loc[common_list, ['spot_number', 'farm_name', 'municipality', 'ph', 'k', 'p', 'n']]

    # Count by location and NPK availability
    print(f"\nRecords by municipality:")
    print(mapping_df['municipality'].value_counts())

    print(f"\nRecords by farm:")
    print(mapping_df['farm_name'].value_counts())

    # NPK coverage by location
    print(f"\nNPK coverage by municipality:")
    for municipality in mapping_df['municipality'].unique():
        muni_df = mapping_df[mapping_df['municipality'] == municipality]
        ph_count = muni_df['ph'].notna().sum()
        k_count = muni_df['k'].notna().sum()
        p_count = muni_df['p'].notna().sum()
        n_count = muni_df['n'].notna().sum()

        print(f"\n  {municipality} ({len(muni_df)} records):")
        print(f"    pH: {ph_count} ({ph_count/len(muni_df)*100:.1f}%)")
        print(f"    K: {k_count} ({k_count/len(muni_df)*100:.1f}%)")
        print(f"    P: {p_count} ({p_count/len(muni_df)*100:.1f}%)")
        print(f"    N: {n_count} ({n_count/len(muni_df)*100:.1f}%)")

    # Save mapping report
    output_path = REPO_ROOT / "data_prepreocessing" / "data" / "combined" / "uuid_mapping_report.csv"
    mapping_df.to_csv(output_path)
    print(f"\n[OK] Mapping report saved to: {output_path}")


def main():
    """Main validation pipeline"""
    print("="*80)
    print("DATA MAPPING & VALIDATION PIPELINE")
    print("Source of Truth: final_merged_data_sorted.csv")
    print("Target: combined_field_data.csv")
    print("="*80)

    # Load data
    source_df, target_df = load_data()

    # Validate UUID mapping
    common_uuids = validate_uuid_mapping(source_df, target_df)

    if not common_uuids:
        print("\n[ERROR] No common UUIDs found between files!")
        sys.exit(1)

    # Compare common columns
    mismatches = compare_common_columns(source_df, target_df, common_uuids)

    # Analyze NPK data
    analyze_npk_data(source_df)

    # Generate mapping report
    generate_mapping_report(source_df, target_df, common_uuids)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
