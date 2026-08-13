def read_file_content(file_path: str) -> dict:
    """
    Read a file and return its contents as a dict of {line_number: content}.

    Returns:
        {
            'total_lines': int,
            'lines': {1: 'first line', 2: 'second line', ...}
        }
    """
    with open(file_path, 'r') as f:
        raw_lines = f.readlines()

    lines = {i: line.rstrip('\n') for i, line in enumerate(raw_lines, start=1)}
    return {
        'total_lines': len(raw_lines),
        'lines': lines,
    }


def replace_file_content(file_path: str, from_lno: int, to_lno: int, replace_with: str) -> dict:
    """
    Replace lines [from_lno, to_lno] (1-indexed, inclusive) in a file with
    the content in replace_with.

    Returns:
        {
            'effective_from': int,        # first line affected (== from_lno)
            'lines_added': int,           # number of new lines written in place of the range
            'lines_removed': int,         # number of old lines that were replaced
            'extra_added_removed': int    # lines_added - lines_removed
                                           # e.g. 4 lines -> 2 lines => -2
                                           # e.g. 4 lines -> 10 lines => 6
        }
    """
    if from_lno < 1 or to_lno < from_lno:
        raise ValueError("Invalid line range: from_lno must be >= 1 and <= to_lno")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    if from_lno > total_lines:
        raise ValueError(f"from_lno ({from_lno}) exceeds file length ({total_lines})")

    # Clamp to_lno to the actual file length in case caller passes something too large
    to_lno_clamped = min(to_lno, total_lines)

    # Figure out the line-ending style used in the file (default to '\n')
    line_ending = '\n'
    if lines and not lines[-1].endswith('\n'):
        # last physical line has no trailing newline
        pass
    if lines:
        for ln in lines:
            if ln.endswith('\n'):
                line_ending = '\n'
                break

    # Split replacement text into individual lines, re-attaching newlines
    new_lines = replace_with.splitlines()
    new_lines = [ln + line_ending for ln in new_lines] if new_lines else []

    # If the original last line in the file had no trailing newline and we're
    # replacing through the end of file, strip the newline from our last new line too
    if to_lno_clamped == total_lines and lines and not lines[-1].endswith('\n') and new_lines:
        new_lines[-1] = new_lines[-1].rstrip('\n')

    lines_removed = to_lno_clamped - from_lno + 1
    lines_added = len(new_lines)

    # Perform the replacement (convert to 0-indexed slice)
    lines[from_lno - 1: to_lno_clamped] = new_lines

    with open(file_path, 'w') as f:
        f.writelines(lines)

    return {
        'effective_from': from_lno,
        'lines_added': lines_added,
        'lines_removed': lines_removed,
        'extra_added_removed': lines_added - lines_removed,
    }