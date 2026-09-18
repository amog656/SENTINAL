import re


def normalize_line_endings(text: str) -> str:
    return text.replace('\r\n', '\n').replace('\r', '\n')


def remove_excessive_blank_lines(text: str, max_blank: int = 2) -> str:
    pattern = r'\n{' + str(max_blank + 1) + r',}'
    return re.sub(pattern, '\n' * max_blank, text)


def remove_repeated_whitespace(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = re.sub(r'[ \t]+', ' ', line)
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def trim_whitespace(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    return '\n'.join(cleaned_lines).strip()


def clean_text(text: str) -> str:
    if not text:
        return ""
    
    text = normalize_line_endings(text)
    text = remove_excessive_blank_lines(text, max_blank=2)
    text = remove_repeated_whitespace(text)
    text = trim_whitespace(text)
    
    return text


def clean_text_preserve_structure(text: str) -> str:
    if not text:
        return ""
    
    text = normalize_line_endings(text)
    
    lines = text.split('\n')
    cleaned_lines = []
    blank_count = 0
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append('')
        else:
            blank_count = 0
            line = re.sub(r'[ \t]+', ' ', line)
            cleaned_lines.append(line.strip())
    
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    while cleaned_lines and not cleaned_lines[0]:
        cleaned_lines.pop(0)
    
    return '\n'.join(cleaned_lines)