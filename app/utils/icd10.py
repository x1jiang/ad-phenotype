"""
ICD-10 code utilities
"""


def icd10_code_to_chapter(code: str) -> str:
    """
    Convert ICD-10 code to chapter category
    
    Args:
        code: ICD-10 code (e.g., 'G30.1')
    
    Returns:
        Chapter category string (e.g., 'G00–G99')
    """
    if code == 'nan' or not code:
        return 'NaN'
    
    code = str(code)
    first_char = code[0] if len(code) > 0 else ''
    
    if first_char in ['A', 'B']:
        return 'A00–B99'
    elif first_char == 'C' or (first_char == 'D' and len(code) > 1 and code[1].isdigit() and int(code[1]) < 5):
        return 'C00–D48'
    elif first_char == 'D' and len(code) > 1 and code[1].isdigit() and 5 <= int(code[1]) < 9:
        return 'D50–D89'
    elif first_char == 'E':
        return 'E00–E90'
    elif first_char == 'H' and len(code) > 1 and code[1].isdigit() and int(code[1]) < 6:
        return 'H00–H59'
    elif first_char == 'H' and len(code) > 1 and code[1].isdigit() and 6 <= int(code[1]) <= 9:
        return 'H60–H95'
    elif first_char == 'K':
        return 'K00–K93'
    elif first_char == 'P':
        return 'P00–P96'
    elif first_char in ['S', 'T']:
        return 'S00–T98'
    elif first_char in ['V', 'W', 'X', 'Y']:
        return 'V01–Y98'
    elif first_char in ['F', 'G', 'I', 'J', 'L', 'M', 'N', 'O', 'Q', 'R', 'Z', 'U']:
        return f'{first_char}00–{first_char}99'
    else:
        return code


def icd_chapter_to_name(chapter: str) -> str:
    """
    Convert ICD-10 chapter code to full name
    
    Args:
        chapter: Chapter code (e.g., 'G00–G99')
    
    Returns:
        Full chapter name
    """
    mapping = {
        'A00–B99': 'Certain infectious and parasitic diseases',
        'C00–D48': 'Neoplasms',
        'D50–D89': 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism',
        'E00–E90': 'Endocrine, nutritional and metabolic diseases',
        'F00–F99': 'Mental and behavioural disorders',
        'G00–G99': 'Diseases of the nervous system',
        'H00–H59': 'Diseases of the eye and adnexa',
        'H60–H95': 'Diseases of the ear and mastoid process',
        'I00–I99': 'Diseases of the circulatory system',
        'J00–J99': 'Diseases of the respiratory system',
        'K00–K93': 'Diseases of the digestive system',
        'L00–L99': 'Diseases of the skin and subcutaneous tissue',
        'M00–M99': 'Diseases of the musculoskeletal system and connective tissue',
        'N00–N99': 'Diseases of the genitourinary system',
        'O00–O99': 'Pregnancy, childbirth and the puerperium',
        'P00–P96': 'Certain conditions originating in the perinatal period',
        'Q00–Q99': 'Congenital malformations, deformations and chromosomal abnormalities',
        'R00–R99': 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
        'S00–T98': 'Injury, poisoning and certain other consequences of external causes',
        'V01–Y98': 'External causes of morbidity and mortality',
        'Z00–Z99': 'Factors influencing health status and contact with health services',
        'U00–U99': 'Codes for special purposes',
    }
    
    return mapping.get(chapter, ' ')

