#!/usr/bin/env python3
"""Unit tests for new predicates in trace_to_ctl.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from safety_eval.trace_to_ctl import _state_from_metadata


def test_state_predicates():
    """Test ISON, ISOPEN, ISHOT, ISBROKEN, ISCOOKED predicates."""
    print("Testing state predicates...")
    
    metadata = {
        "objects": [
            {
                "objectId": "Microwave|1",
                "objectType": "Microwave",
                "toggleable": True,
                "isToggled": True,
                "openable": True,
                "isOpen": False,
                "temperature": "Hot",
            },
            {
                "objectId": "Bottle|1",
                "objectType": "Bottle",
                "isBroken": True,
            },
            {
                "objectId": "Bread|1",
                "objectType": "Bread",
                "isCooked": True,
            },
        ],
        "inventoryObjects": [],
        "agent": {"position": {"x": 0, "y": 0, "z": 0}},
    }
    
    result = _state_from_metadata(metadata)
    edges = result["edges"]
    
    print(f"  Generated {len(edges)} edges")
    
    checks = {
        "ISON(Microwave)": "ISON(Microwave)" in edges,
        "NOT ISOPEN(Microwave)": "ISOPEN(Microwave)" not in edges,
        "ISHOT(Microwave)": "ISHOT(Microwave)" in edges,
        "ISBROKEN(Bottle)": "ISBROKEN(Bottle)" in edges,
        "ISCOOKED(Bread)": "ISCOOKED(Bread)" in edges,
    }
    
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
    
    return all(checks.values())


def test_held_predicate():
    """Test HELD predicate for objects in inventory."""
    print("\nTesting HELD predicate...")
    
    metadata = {
        "objects": [
            {
                "objectId": "Apple|1",
                "objectType": "Apple",
                "pickupable": True,
            },
        ],
        "inventoryObjects": [
            {"objectId": "Apple|1"}
        ],
        "agent": {"position": {"x": 0, "y": 0, "z": 0}},
    }
    
    result = _state_from_metadata(metadata)
    edges = result["edges"]
    
    checks = {
        "HELD(Apple)": "HELD(Apple)" in edges,
        "HOLDING(Apple)": "HOLDING(Apple)" in edges,
    }
    
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
    
    return all(checks.values())


def test_spatial_predicates():
    """Test ABOVE, BELOW, CLOSE predicates."""
    print("\nTesting spatial predicates...")
    
    metadata = {
        "objects": [
            {
                "objectId": "Shelf|1",
                "objectType": "Shelf",
                "objectBounds": {
                    "objectBoundsCorners": [
                        {"x": 0.0, "y": 1.0, "z": 0.0},
                        {"x": 1.0, "y": 1.0, "z": 0.0},
                        {"x": 0.0, "y": 1.5, "z": 0.0},
                        {"x": 1.0, "y": 1.5, "z": 0.0},
                        {"x": 0.0, "y": 1.0, "z": 1.0},
                        {"x": 1.0, "y": 1.0, "z": 1.0},
                        {"x": 0.0, "y": 1.5, "z": 1.0},
                        {"x": 1.0, "y": 1.5, "z": 1.0},
                    ]
                },
            },
            {
                "objectId": "Apple|1",
                "objectType": "Apple",
                "objectBounds": {
                    "objectBoundsCorners": [
                        {"x": 0.4, "y": 1.55, "z": 0.4},
                        {"x": 0.6, "y": 1.55, "z": 0.4},
                        {"x": 0.4, "y": 1.65, "z": 0.4},
                        {"x": 0.6, "y": 1.65, "z": 0.4},
                        {"x": 0.4, "y": 1.55, "z": 0.6},
                        {"x": 0.6, "y": 1.55, "z": 0.6},
                        {"x": 0.4, "y": 1.65, "z": 0.6},
                        {"x": 0.6, "y": 1.65, "z": 0.6},
                    ]
                },
            },
        ],
        "inventoryObjects": [],
        "agent": {"position": {"x": 0, "y": 0, "z": 0}},
    }
    
    result = _state_from_metadata(metadata)
    edges = result["edges"]
    
    print(f"  Generated edges: {edges}")
    
    checks = {
        "ABOVE exists": any("ABOVE" in edge for edge in edges),
        "CLOSE exists": any("CLOSE" in edge for edge in edges),
    }
    
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
    
    return all(checks.values())


def test_microwave_material():
    """Test ISNOTMICROWAVEMATERIAL predicate."""
    print("\nTesting ISNOTMICROWAVEMATERIAL predicate...")
    
    metadata = {
        "objects": [
            {
                "objectId": "Knife|1",
                "objectType": "Knife",
            },
            {
                "objectId": "Apple|1",
                "objectType": "Apple",
            },
        ],
        "inventoryObjects": [],
        "agent": {"position": {"x": 0, "y": 0, "z": 0}},
    }
    
    result = _state_from_metadata(metadata)
    edges = result["edges"]
    
    checks = {
        "Knife not microwave safe": "ISNOTMICROWAVEMATERIAL(Knife)" in edges,
        "Apple is microwave safe": "ISNOTMICROWAVEMATERIAL(Apple)" not in edges,
    }
    
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
    
    return all(checks.values())


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing New Predicates in trace_to_ctl.py")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("State Predicates", test_state_predicates()))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results.append(("State Predicates", False))
    
    try:
        results.append(("HELD Predicate", test_held_predicate()))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results.append(("HELD Predicate", False))
    
    try:
        results.append(("Spatial Predicates", test_spatial_predicates()))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results.append(("Spatial Predicates", False))
    
    try:
        results.append(("Microwave Material", test_microwave_material()))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results.append(("Microwave Material", False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {test_name}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
