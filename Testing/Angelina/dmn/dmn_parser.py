"""
DMN Parser - Converts DMN XML files to JSON format
"""

from dmn_loader import DMNImport
import json
import os
from typing import List, Dict, Any, Optional

class DMNParser:
    """
    Parse DMN files and convert to desired format:
    [{ conditions: {...}, outputs: {...} }, ...]
    """
    
    def __init__(self, file_path: str):
        """
        Initialize parser with path to DMN file
        
        Args:
            file_path: Path to .dmn file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"DMN file not found: {file_path}")
            
        self.file_path = file_path
        self.importer = DMNImport(file_path)
        self.model = self.importer.importDMN()
        self.decisions = self.model.decisions
        
    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse all decisions and return rules in desired format
        
        Returns:
            List of dictionaries with 'conditions' and 'outputs' keys
        """
        all_rules = []
        
        for decision in self.decisions:
            decision_name = decision.name
            
            # Skip if no decision table
            if not hasattr(decision, 'decisionTable') or decision.decisionTable is None:
                continue
                
            decision_table = decision.decisionTable
            rules = self._extract_rules_from_table(decision_table, decision_name)
            all_rules.extend(rules)
            
        return all_rules
    
    def _extract_rules_from_table(self, decision_table, decision_name: str) -> List[Dict[str, Any]]:
        """Extract rules from a single decision table"""
        rules = []
        
        # Get input and output info
        inputs = decision_table.inputs
        outputs = decision_table.outputs
        
        # Extract input names
        input_names = []
        for inp in inputs:
            if hasattr(inp, 'inputExpression') and inp.inputExpression:
                input_names.append(inp.inputExpression.text)
            else:
                input_names.append(inp.label or "unknown_input")
        
        # Extract output names
        output_names = [out.name for out in outputs if out.name]
        
        # Process each rule
        for rule_idx, rule in enumerate(decision_table.rules):
            conditions = {}
            outputs_dict = {}
            
            # Process input entries (conditions)
            for i, input_entry in enumerate(rule.inputEntries):
                if i < len(input_names):
                    value = self._clean_value(input_entry.text)
                    # Skip wildcards
                    if value is not None and value != '-':
                        conditions[input_names[i]] = self._parse_value(value)
            
            # Process output entries
            for i, output_entry in enumerate(rule.outputEntries):
                if i < len(output_names):
                    value = self._clean_value(output_entry.text)
                    if value is not None:
                        outputs_dict[output_names[i]] = self._parse_value(value)
            
            # Add metadata
            outputs_dict['_decision'] = decision_name
            
            rules.append({
                'conditions': conditions,
                'outputs': outputs_dict,
                'metadata': {
                    'decision': decision_name,
                    'hit_policy': decision_table.hitPolicy,
                    'rule_index': rule_idx
                }
            })
        
        return rules
    
    def _clean_value(self, value: Optional[str]) -> Optional[str]:
        """Clean up value strings"""
        if value is None:
            return None
        value = value.strip()
        return value if value else None
    
    def _parse_value(self, value: str):
        """Parse string values to appropriate Python types"""
        # Handle quoted strings
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        
        # Handle booleans
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        
        # Handle numbers
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Return as-is for expressions (like ">= 14")
        return value
    
    def parse_and_save(self, output_file: str = 'parsed_rules.json') -> List[Dict[str, Any]]:
        """
        Parse DMN and save results to JSON file
        
        Args:
            output_file: Path to output JSON file
            
        Returns:
            Parsed rules
        """
        rules = self.parse()
        
        with open(output_file, 'w') as f:
            json.dump(rules, f, indent=2)
        
        print(f"✅ Saved {len(rules)} rules to {output_file}")
        return rules
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the parsed DMN
        """
        rules = self.parse()
        
        stats = {
            'total_rules': len(rules),
            'decisions': {},
            'input_fields': set(),
            'output_fields': set()
        }
        
        # Count by decision
        for rule in rules:
            dec = rule['metadata']['decision']
            stats['decisions'][dec] = stats['decisions'].get(dec, 0) + 1
            stats['input_fields'].update(rule['conditions'].keys())
            stats['output_fields'].update(rule['outputs'].keys())
        
        # Convert sets to lists for JSON serialization
        stats['input_fields'] = list(stats['input_fields'])
        stats['output_fields'] = list(stats['output_fields'])
        
        return stats
    
    def print_summary(self):
        """Print a human-readable summary"""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("📊 DMN PARSER SUMMARY")
        print("="*50)
        print(f"File: {self.file_path}")
        print(f"Total rules: {stats['total_rules']}")
        print(f"Total decisions: {len(stats['decisions'])}")
        print(f"Unique input fields: {len(stats['input_fields'])}")
        print(f"Unique output fields: {len(stats['output_fields'])}")
        
        print("\n📋 Rules per decision:")
        for dec, count in stats['decisions'].items():
            print(f"  • {dec}: {count} rules")
        
        print("\n🔍 Input fields:")
        for field in sorted(stats['input_fields'])[:10]:  # Show first 10
            print(f"  • {field}")
        if len(stats['input_fields']) > 10:
            print(f"  ... and {len(stats['input_fields'])-10} more")
        
        print("\n📤 Output fields:")
        for field in sorted(stats['output_fields']):
            print(f"  • {field}")
        
        print("="*50)


# If run directly, parse the DMN file
if __name__ == "__main__":
    # Specify your DMN file
    dmn_file = "modular_ccm_rules.dmn"  # Change this to your actual filename
    
    try:
        # Create parser
        parser = DMNParser(dmn_file)
        
        # Print summary
        parser.print_summary()
        
        # Parse and save to JSON
        rules = parser.parse_and_save("parsed_rules.json")
        
        # Show sample rules
        print("\n📝 Sample Rules (first 3):")
        for i, rule in enumerate(rules[:3]):
            print(f"\n  Rule {i+1}:")
            print(f"    Conditions: {rule['conditions']}")
            print(f"    Outputs: {rule['outputs']}")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please make sure your DMN file is in the same directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")