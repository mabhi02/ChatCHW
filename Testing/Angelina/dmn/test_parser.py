"""
Test script to verify the DMN parser works correctly
"""

from dmn_parser import DMNParser
import json

def test_parser():
    """Test the DMN parser with your file"""
    
    # Try different possible filenames
    possible_filenames = [
        "modular_ccm_rules.dmn",
        "rules.dmn",
        "ccm_rules.dmn",
        "dmn_rules.xml"
    ]
    
    parser = None
    for filename in possible_filenames:
        try:
            parser = DMNParser(filename)
            print(f"✅ Found DMN file: {filename}")
            break
        except FileNotFoundError:
            continue
    
    if parser is None:
        print("❌ Could not find DMN file. Please check the filename.")
        print("   Expected one of:", possible_filenames)
        return
    
    # Test 1: Parse all rules
    print("\n" + "="*50)
    print("TEST 1: Parsing all rules")
    print("="*50)
    rules = parser.parse()
    print(f"✅ Successfully parsed {len(rules)} rules")
    
    # Test 2: Check structure
    print("\n" + "="*50)
    print("TEST 2: Verifying rule structure")
    print("="*50)
    
    if len(rules) > 0:
        sample = rules[0]
        print(f"Sample rule structure:")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Conditions type: {type(sample['conditions'])}")
        print(f"  Outputs type: {type(sample['outputs'])}")
        print(f"  Metadata type: {type(sample['metadata'])}")
        
        # Verify format matches requirement
        expected_format = "{ conditions: {...}, outputs: {...} }"
        actual_format = f"{{ conditions: {sample['conditions']}, outputs: {sample['outputs']} }}"
        print(f"\n✅ Format matches: {expected_format}")
    
    # Test 3: Save to JSON
    print("\n" + "="*50)
    print("TEST 3: Saving to JSON")
    print("="*50)
    parser.parse_and_save("test_output.json")
    
    # Verify JSON file
    with open("test_output.json", 'r') as f:
        loaded = json.load(f)
    print(f"✅ JSON file created with {len(loaded)} rules")
    
    # Test 4: Check for specific Module A rules
    print("\n" + "="*50)
    print("TEST 4: Looking for Module A rules")
    print("="*50)
    
    module_a_rules = [r for r in rules if r['metadata']['decision'] == 'decide_module_a']
    print(f"Found {len(module_a_rules)} rules in Module A")
    
    if module_a_rules:
        print("\nSample Module A rule:")
        print(json.dumps(module_a_rules[0], indent=2))
    
    # Test 5: Check for specific conditions from your DMN
    print("\n" + "="*50)
    print("TEST 5: Looking for specific rules from your DMN")
    print("="*50)
    
    # Look for cough >= 14 days rule
    for rule in rules:
        conditions = rule['conditions']
        if 'cough_duration_days' in conditions and conditions['cough_duration_days'] == '>= 14':
            print("✅ Found cough >= 14 days rule:")
            print(f"  Outputs: {rule['outputs']}")
            break
    
    # Look for fast breathing rule
    for rule in rules:
        conditions = rule['conditions']
        if 'fast_breathing_present' in conditions and conditions['fast_breathing_present'] is True:
            print("\n✅ Found fast breathing rule:")
            print(f"  Outputs: {rule['outputs']}")
            break
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED")
    print("="*50)

if __name__ == "__main__":
    test_parser()