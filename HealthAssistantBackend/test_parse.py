def parse_examination_text(text):
    """
    Parse examination text to extract the examination description and findings.
    Handles three scenarios:
    1. Clear examination procedure with findings
    2. No examination information available
    3. Partial/informal examination information
    
    Args:
        text (str): The examination text to parse
    
    Returns:
        tuple: (examination_description, list_of_findings)
    """
    print("Input text:", repr(text))
    
    # Process empty or very short responses
    if not text or len(text.strip()) < 10:
        print("Case: Empty or very short response")
        examination = "No examination information provided in the medical guide."
        findings = [
            "No specific finding available - refer to higher facility",
            "Unable to perform examination based on medical guide",
            "Need additional clinical assessment",
            "Consider general observation only"
        ]
        return examination, findings
    
    # Common phrases indicating no information is available
    no_info_phrases = [
        "does not provide", "no information", "no examination", 
        "no specific examination", "doesn't provide", "isn't provided",
        "the guide doesn't provide", "not available in the guide",
        "not included in the guide", "not mentioned in the guide",
        "doesn't contain", "does not mention"
    ]
    
    # SCENARIO 1: Check if the response indicates no examination information is available
    if any(phrase in text.lower() for phrase in no_info_phrases):
        print("Case: No information available (matched phrases)")
        # Return a standardized "No examination" response with default findings
        examination = text.strip()
        findings = [
            "No specific finding available - refer to higher facility",
            "Unable to perform examination based on medical guide",
            "Need additional clinical assessment",
            "Consider general observation only"
        ]
        return examination, findings
    
    # SCENARIO 2: Check if there's some useful information but not a formal procedure
    if ("relevant information" in text.lower() or "some information" in text.lower() or 
        "general guidance" in text.lower() or "may be helpful" in text.lower() or
        not any(line.startswith('#') or line.startswith('#:') for line in text.strip().split('\n'))):
        
        print("Case: Partial/informal information")
        # Extract the useful information
        examination = "PARTIAL INFORMATION (Not a formal examination procedure):\n" + text.strip()
        findings = [
            "Consider referring to medical professional for proper examination",
            "Use this information as supplementary guidance only",
            "Document observations based on general assessment",
            "Consult with supervisor about next steps"
        ]
        return examination, findings
    
    # SCENARIO 3: Regular parsing for normal examination text with findings
    print("Case: Full examination with findings")
    # Split the text by lines
    lines = text.strip().split('\n')
    
    # Extract examination description (everything before the first finding)
    examination_lines = []
    findings = []
    
    in_examination = True
    
    for line in lines:
        line = line.strip()
        # Check for both '#:' and '#' formats
        if line.startswith('#:') or (line.startswith('#') and not line.startswith('#:')):
            in_examination = False
            # Extract the finding text by removing the delimiter
            if line.startswith('#:'):
                finding = line[2:].strip()
            else:
                finding = line[1:].strip()
            findings.append(finding)
        elif in_examination and line:
            examination_lines.append(line)
    
    # Join the examination lines
    examination = '\n'.join(examination_lines)
    
    # If we didn't find any findings but have examination text, create default findings
    if not findings and examination:
        print("Case: Examination text but no findings - creating default findings")
        findings = [
            "Normal finding",
            "Abnormal finding requiring further assessment",
            "Inconclusive finding - may need additional tests",
            "Unable to determine based on current examination"
        ]
    # If we don't have examination text or findings, provide a fallback response
    elif not findings and not examination:
        print("Case: No examination text and no findings - providing fallback")
        examination = "The response from the medical guide was incomplete or invalid."
        findings = [
            "No specific finding available - refer to higher facility",
            "Unable to perform examination based on medical guide",
            "Need additional clinical assessment",
            "Consider general observation only"
        ]
    
    return examination, findings

def test_parse():
    # Test case 1: No information from guide
    result1 = parse_examination_text("The guide doesn't provide that information.")
    print("Test 1 - No information:")
    print("Examination:", result1[0])
    print("Findings:", result1[1])
    print()
    
    # Test case 2: Partial information
    result2 = parse_examination_text("While the medical guide does not provide a formal examination procedure, here is some relevant information that may be helpful: Monitor for blood in stool.")
    print("Test 2 - Partial information:")
    print("Examination:", result2[0])
    print("Findings:", result2[1])
    print()
    
    # Test case 3: Full examination
    test_text = """Examination of abdomen
Palpate the abdomen gently.
#: Tenderness in the right lower quadrant
#: No tenderness
#: Mass felt in the abdomen
#: Rigidity of abdominal muscles"""
    result3 = parse_examination_text(test_text)
    print("Test 3 - Full examination:")
    print("Examination:", result3[0])
    print("Findings:", result3[1])

if __name__ == "__main__":
    test_parse() 