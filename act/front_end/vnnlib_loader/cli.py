#!/usr/bin/env python3
"""
Command-Line Interface for VNNLIB Category Management.

Provides CLI tools for exploring VNN-COMP categories, downloading benchmarks,
parsing VNNLIB files, and managing ONNX models.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import argparse
from typing import Optional

from act.front_end.vnnlib_loader.category_mapping import (
    CATEGORY_MAPPING,
    get_category_info,
    list_categories,
    list_categories_by_type,
    get_all_types,
    search_categories,
    find_category_name,
    get_summary_statistics,
)


def print_category_list(category_type: Optional[str] = None):
    """
    Print list of available VNNLIB categories.
    
    Args:
        category_type: If provided, only show categories of this type
    """
    if category_type:
        categories = list_categories_by_type(category_type)
        print(f"\n{'='*100}")
        print(f"VNNLIB CATEGORIES - TYPE: {category_type}")
        print(f"{'='*100}")
    else:
        categories = list_categories()
        print(f"\n{'='*100}")
        print(f"ALL VNNLIB CATEGORIES ({len(categories)})")
        print(f"{'='*100}")
    
    if not categories:
        print(f"No categories found for type: {category_type}")
        return
    
    # Group by type for display
    by_type = {}
    for cat_name in categories:
        info = CATEGORY_MAPPING[cat_name]
        cat_type = info['type']
        if cat_type not in by_type:
            by_type[cat_type] = []
        by_type[cat_type].append((cat_name, info))
    
    # Print by type
    for cat_type in sorted(by_type.keys()):
        items = by_type[cat_type]
        print(f"\n{cat_type.upper()} ({len(items)} categories)")
        print('-' * 100)
        
        for cat_name, info in sorted(items, key=lambda x: x[0]):
            print(f"  {cat_name:30s} ({info['year']}) - {info['description']}")


def print_category_detail(category_name: str):
    """
    Print detailed information about a specific category.
    
    Args:
        category_name: Name of the category (case-insensitive)
    """
    try:
        actual_name = find_category_name(category_name)
        info = get_category_info(actual_name)
        
        print(f"\n{'='*100}")
        print(f"CATEGORY: {actual_name}")
        print(f"{'='*100}")
        print(f"Type: {info['type']}")
        print(f"Year: {info['year']}")
        print(f"Description: {info['description']}")
        
        print(f"\nModel Information:")
        print(f"  • Models: {info['models']}")
        print(f"  • Properties: {info['properties']}")
        
        print(f"\nDimensions:")
        print(f"  • Input: {info['input_dim']}")
        print(f"  • Output: {info['output_dim']}")
        
        print(f"{'='*100}\n")
        
    except ValueError as e:
        print(f"Error: {e}")


def print_summary_statistics():
    """Print summary statistics about VNNLIB categories."""
    stats = get_summary_statistics()
    
    print(f"\n{'='*100}")
    print(f"VNNLIB CATEGORIES SUMMARY")
    print(f"{'='*100}")
    print(f"Total Categories: {stats['total_categories']}")
    print(f"Category Types: {stats['total_types']}")
    print(f"Year Range: {stats['oldest_year']} - {stats['newest_year']}")
    
    print(f"\nCategories by Type:")
    for cat_type, count in sorted(stats['categories_by_type'].items()):
        print(f"  • {cat_type:30s}: {count:2d} categories")
    
    print(f"{'='*100}\n")


def main():
    """Main CLI entry point for VNNLIB category management."""
    parser = argparse.ArgumentParser(
        description="VNNLIB Category Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # List and search commands
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available VNNLIB categories"
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        help="Show categories of specific type (e.g., image_classification, control, collision_avoidance)"
    )
    parser.add_argument(
        "--info", "-i",
        type=str,
        metavar="CATEGORY",
        help="Show detailed information about a specific category"
    )
    parser.add_argument(
        "--search", "-s",
        type=str,
        help="Search for categories by name"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics"
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        dest="list_types",
        help="List all available category types"
    )
    
    # Download and management commands (placeholders for future implementation)
    parser.add_argument(
        "--download",
        type=str,
        metavar="CATEGORY",
        help="Download a VNNLIB category (ONNX + VNNLIB files) [NOT IMPLEMENTED]"
    )
    parser.add_argument(
        "--list-downloads",
        action="store_true",
        dest="list_downloads",
        help="List all downloaded categories [NOT IMPLEMENTED]"
    )
    parser.add_argument(
        "--load",
        nargs=2,
        metavar=("CATEGORY", "INSTANCE_ID"),
        help="Load a specific instance from a category [NOT IMPLEMENTED]"
    )
    
    # VNNLIB-specific commands (placeholders)
    parser.add_argument(
        "--parse-vnnlib",
        type=str,
        metavar="FILE",
        dest="parse_vnnlib",
        help="Parse a VNNLIB file and show constraints [NOT IMPLEMENTED]"
    )
    parser.add_argument(
        "--validate-vnnlib",
        type=str,
        metavar="FILE",
        dest="validate_vnnlib",
        help="Validate VNNLIB file syntax [NOT IMPLEMENTED]"
    )
    parser.add_argument(
        "--instance-info",
        nargs=2,
        metavar=("CATEGORY", "INSTANCE_ID"),
        dest="instance_info",
        help="Show details about a specific instance [NOT IMPLEMENTED]"
    )
    
    args = parser.parse_args()
    
    # Handle commands
    if args.list:
        print_category_list()
    
    elif args.type:
        try:
            print_category_list(category_type=args.type)
        except ValueError as e:
            print(f"Error: {e}")
            print(f"\nAvailable types: {', '.join(get_all_types())}")
    
    elif args.info:
        print_category_detail(args.info)
    
    elif args.search:
        matches = search_categories(args.search)
        if matches:
            print(f"\nCategories matching '{args.search}':")
            for name in sorted(matches):
                info = CATEGORY_MAPPING[name]
                print(f"  • {name:30s} ({info['type']}) - {info['description']}")
        else:
            print(f"No categories found matching '{args.search}'")
    
    elif args.summary:
        print_summary_statistics()
    
    elif args.list_types:
        types = get_all_types()
        print(f"\nAvailable Category Types ({len(types)}):")
        for cat_type in sorted(types):
            # Count categories of this type
            count = len(list_categories_by_type(cat_type))
            print(f"  • {cat_type:30s} ({count} categories)")
    
    elif args.download:
        print(f"\n⚠️  Download functionality not yet implemented.")
        print(f"    Category: {args.download}")
        print(f"\n    This feature will download:")
        print(f"    • ONNX model files")
        print(f"    • VNNLIB property files")
        print(f"    • instances.csv mapping file")
        print(f"\n    Coming soon!")
    
    elif args.list_downloads:
        print(f"\n⚠️  List downloads functionality not yet implemented.")
        print(f"    This will show all downloaded VNNLIB categories.")
    
    elif args.load:
        category, instance_id = args.load
        print(f"\n⚠️  Load functionality not yet implemented.")
        print(f"    Category: {category}")
        print(f"    Instance: {instance_id}")
    
    elif args.parse_vnnlib:
        print(f"\n⚠️  VNNLIB parsing not yet implemented.")
        print(f"    File: {args.parse_vnnlib}")
    
    elif args.validate_vnnlib:
        print(f"\n⚠️  VNNLIB validation not yet implemented.")
        print(f"    File: {args.validate_vnnlib}")
    
    elif args.instance_info:
        category, instance_id = args.instance_info
        print(f"\n⚠️  Instance info not yet implemented.")
        print(f"    Category: {category}")
        print(f"    Instance: {instance_id}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
