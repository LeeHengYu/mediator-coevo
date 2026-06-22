# Task Instruction

Complete the project proposal HWPX document by following these steps precisely:

## Step 1: Inspect the workspace
```bash
ls /root/
cat /root/project_proposal.json
```

## Step 2: Run the following Python script to perform the entire transformation

Create and execute `/root/fill_proposal.py` with this content:

```python
import json
import os
import re
import shutil
import zipfile

# Paths
TEMPLATE = '/root/project_proposal_template.hwpx'
OUTPUT = '/root/project_proposal_ready.hwpx'
EXTRACT_DIR = '/root/hwpx_extracted'

# Clean up any previous extraction
if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)

# Step 1: Extract the HWPX (ZIP) archive
with zipfile.ZipFile(TEMPLATE, 'r') as zf:
    zf.extractall(EXTRACT_DIR)

# Step 2: Load JSON values
with open('/root/project_proposal.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Flatten nested JSON if needed (handle both flat and nested structures)
def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

flat_data = flatten(data)
print("JSON keys (flat):", list(flat_data.keys()))
print("JSON data:", json.dumps(data, ensure_ascii=False, indent=2))

# Step 3: Process all XML files in the extracted directory
def compute_month_span(date_range_str):
    """Given a date range like '2025.01 ~ 2025.03', compute the month span."""
    match = re.search(r'(\d{4})\.(\d{2})\s*~\s*(\d{4})\.(\d{2})', date_range_str)
    if match:
        y1, m1, y2, m2 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        months = (y2 - y1) * 12 + (m2 - m1)
        # If same month, count as 1 month
        if months == 0:
            months = 1
        return f"({months}개월)"
    return None

for root, dirs, files in os.walk(EXTRACT_DIR):
    for fname in files:
        fpath = os.path.join(root, fname)
        if fname.endswith('.xml'):
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Replace placeholders {{key}} with values from JSON
            # First try direct keys, then try nested dot-notation keys
            def replace_placeholder(m):
                key = m.group(1)
                # Try direct lookup in original data
                val = data.get(key)
                if val is None:
                    # Try flat lookup
                    val = flat_data.get(key)
                if val is None:
                    # Try nested access
                    parts = key.split('.')
                    obj = data
                    for p in parts:
                        if isinstance(obj, dict):
                            obj = obj.get(p)
                        else:
                            obj = None
                            break
                    val = obj
                if val is not None:
                    val_str = str(val)
                    # Normalize budget: remove commas but keep currency symbol
                    if '예산' in key or 'budget' in key.lower() or (',' in val_str and re.search(r'[₩$€]', val_str)):
                        val_str = val_str.replace(',', '')
                    # Also check if the value itself has commas and looks like a number with currency
                    return val_str
                print(f"WARNING: No value found for placeholder: {{{{{key}}}}}")
                return m.group(0)  # Leave as-is if not found (should not happen)
            
            # Handle placeholders that might be split across XML tags
            # First, try to find and fix split placeholders
            # Pattern: {{ might be in one <hp:t> and }} in another
            # Strategy: collect all text, replace, then put back
            
            # Simple approach: replace on the raw XML string
            # This works when placeholders are not split across tags
            content = re.sub(r'\{\{([^}]+)\}\}', replace_placeholder, content)
            
            # If placeholders are split across tags, handle that:
            # Look for patterns like >{{</hp:t>...<hp:t>key}}<
            # Consolidate by removing intermediate tags
            split_pattern = r'(\{\{[^}]*?)(<[^>]*>)(.*?)(</[^>]*>)([^}]*?\}\})'
            while re.search(r'\{\{', content) and re.search(split_pattern, content):
                # Try to consolidate split placeholders
                def consolidate(m):
                    full = m.group(0)
                    # Remove XML tags between {{ and }}
                    text_only = re.sub(r'<[^>]+>', '', full)
                    return text_only
                content_new = re.sub(split_pattern, consolidate, content)
                if content_new == content:
                    break
                content = content_new
                content = re.sub(r'\{\{([^}]+)\}\}', replace_placeholder, content)
            
            # More aggressive split-tag handling: remove tags between {{ and }}
            if '{{' in content:
                # Find all {{ ... }} that span across tags
                content = re.sub(r'\{\{((?:(?!\}\}).)*?)\}\}', 
                    lambda m: re.sub(r'<[^>]+>', '', m.group(0)),
                    content, flags=re.DOTALL)
                content = re.sub(r'\{\{([^}]+)\}\}', replace_placeholder, content)
            
            # Normalize any budget values that have commas (catch-all)
            # Look for patterns like ₩1,500,000,000 and remove commas
            content = re.sub(r'(₩)(\d{1,3}(?:,\d{3})+)', 
                lambda m: m.group(1) + m.group(2).replace(',', ''), content)
            
            # Add month spans after phase lines (단계N)
            # Look for lines containing 단계1, 단계2, 단계3 with date ranges
            # and append (N개월) after the phase description
            def add_month_span(line_content):
                # Find 단계N patterns with date ranges in the same text context
                phase_pattern = r'(단계\d[^<]*?)(\d{4}\.\d{2}\s*~\s*\d{4}\.\d{2})'
                match = re.search(phase_pattern, line_content)
                if match:
                    date_range = match.group(2)
                    span = compute_month_span(date_range)
                    if span and span not in line_content:
                        # Append month span after the date range
                        line_content = line_content.replace(date_range, date_range + ' ' + span, 1)
                return line_content
            
            # Process phase lines - work on text within <hp:t> tags
            # But we need to handle the case where 단계 and date might be in same or different tags
            # First, let's check the overall content for 단계 patterns
            if '단계' in content:
                # Try to add month spans to lines containing 단계
                # Work on the full content string to handle cross-tag scenarios
                phase_matches = list(re.finditer(r'단계\d', content))
                for pm in phase_matches:
                    # Find the surrounding context (up to next paragraph boundary)
                    start = max(0, pm.start() - 200)
                    end = min(len(content), pm.end() + 500)
                    context = content[start:end]
                    date_match = re.search(r'(\d{4}\.\d{2}\s*~\s*\d{4}\.\d{2})', context[pm.start()-start:])
                    if date_match:
                        date_range = date_match.group(1)
                        span = compute_month_span(date_range)
                        if span:
                            # Find this date range in the actual content near this phase
                            search_start = pm.start()
                            search_end = min(len(content), pm.end() + 500)
                            region = content[search_start:search_end]
                            dr_pos = region.find(date_range)
                            if dr_pos >= 0:
                                # Check if month span already present right after
                                after_dr = region[dr_pos+len(date_range):dr_pos+len(date_range)+20]
                                if '개월' not in after_dr:
                                    # Insert the month span
                                    # Find the end of the <hp:t> tag containing the date range end
                                    abs_pos = search_start + dr_pos + len(date_range)
                                    # Check if we're inside a tag
                                    next_close = content.find('</hp:t>', abs_pos)
                                    next_open = content.find('<', abs_pos)
                                    if next_close >= 0 and (next_open == -1 or next_close <= next_open + 10):
                                        # Insert before the closing tag
                                        content = content[:abs_pos] + ' ' + span + content[abs_pos:]
                                    else:
                                        # Insert directly after the date range
                                        content = content[:abs_pos] + ' ' + span + content[abs_pos:]
            
            # Remove layout cache (lineSegArray / linesegarray) from modified paragraphs
            # To be safe, remove ALL lineSegArray elements since we may have modified text
            if content != original_content:
                # Remove <hp:lineSegArray>...</hp:lineSegArray> (case insensitive for tag name)
                content = re.sub(r'<hp:lineSegArray[^>]*>.*?</hp:lineSegArray>', '', content, flags=re.DOTALL|re.IGNORECASE)
                content = re.sub(r'<hp:linesegarray[^>]*>.*?</hp:linesegarray>', '', content, flags=re.DOTALL|re.IGNORECASE)
                # Also handle self-closing variants
                content = re.sub(r'<hp:lineSegArray[^/]*/>', '', content, flags=re.IGNORECASE)
                content = re.sub(r'<hp:linesegarray[^/]*/>', '', content, flags=re.IGNORECASE)
            
            # Verify no placeholders remain
            remaining = re.findall(r'\{\{[^}]+\}\}', content)
            if remaining:
                print(f"WARNING: Remaining placeholders in {fpath}: {remaining}")
            
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            if content != original_content:
                print(f"Modified: {fpath}")

# Step 4: Repackage as HWPX
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

# Create the ZIP with mimetype first and uncompressed
with zipfile.ZipFile(OUTPUT, 'w') as zf:
    # Add mimetype first, uncompressed
    mimetype_path = os.path.join(EXTRACT_DIR, 'mimetype')
    if os.path.exists(mimetype_path):
        zf.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
    
    # Add all other files
    for root_dir, dirs, files_list in os.walk(EXTRACT_DIR):
        for fname in sorted(files_list):
            fpath = os.path.join(root_dir, fname)
            arcname = os.path.relpath(fpath, EXTRACT_DIR)
            if arcname == 'mimetype':
                continue  # Already added
            zf.write(fpath, arcname, compress_type=zipfile.ZIP_DEFLATED)

print(f"\nOutput written to: {OUTPUT}")
print(f"Output size: {os.path.getsize(OUTPUT)} bytes")

# Final verification: check no placeholders remain
with zipfile.ZipFile(OUTPUT, 'r') as zf:
    for name in zf.namelist():
        if name.endswith('.xml'):
            xml_content = zf.read(name).decode('utf-8')
            remaining = re.findall(r'\{\{[^}]+\}\}', xml_content)
            if remaining:
                print(f"ERROR: Placeholders remain in {name}: {remaining}")
            # Check for month spans
            if '단계' in xml_content:
                print(f"Phase content in {name}:")
                for line in xml_content.split('\n'):
                    if '단계' in line:
                        # Extract text content
                        texts = re.findall(r'>([^<]*단계[^<]*)<', line)
                        for t in texts:
                            print(f"  {t}")
                # Also check for 개월
                spans = re.findall(r'\(\d+개월\)', xml_content)
                print(f"  Month spans found: {spans}")
    print("\nVerification complete.")
```

Run the script:
```bash
python3 /root/fill_proposal.py
```

## Step 3: Verify the output
```bash
ls -la /root/project_proposal_ready.hwpx
python3 -c "
import zipfile, re
with zipfile.ZipFile('/root/project_proposal_ready.hwpx', 'r') as zf:
    print('Archive entries:', zf.namelist()[:5], '...')
    for name in zf.namelist():
        if name.endswith('.xml'):
            c = zf.read(name).decode('utf-8')
            placeholders = re.findall(r'\{\{[^}]+\}\}', c)
            if placeholders:
                print(f'FAIL: {name} has placeholders: {placeholders}')
            if '개월' in c:
                spans = re.findall(r'\(\d+개월\)', c)
                print(f'{name}: month spans = {spans}')
            # Check budget has no commas
            if '₩' in c:
                budgets = re.findall(r'₩[\d,]+', c)
                print(f'{name}: budget values = {budgets}')
                for b in budgets:
                    if ',' in b:
                        print(f'FAIL: Budget still has commas: {b}')
print('Verification done')
"
```

## Important Notes
- The `mimetype` file MUST be the first ZIP entry and stored uncompressed (ZIP_STORED).
- ALL `<hp:lineSegArray>` elements in modified XML files must be removed to prevent text overlap.
- Budget values must have commas removed but keep the ₩ symbol (e.g., ₩1500000000).
- Each 단계 (phase) line must have a parenthesized month span appended based on the date range in that line.
- No `{{...}}` placeholders may remain in the output.
- All Korean labels and static note lines must remain unchanged.

# Executor Policy

---
name: executor
description: Portable executor policy for workflow, verification, resource use, and failure handling across task runtimes.
---

## Executor Policy

Use this skill as execution policy, not as domain-specific task knowledge. When
task-local curated skills or resources are available, prefer them for domain
details and use this policy for workflow control.

## Task Execution

1. Read the task instruction, task resources, and verifier contract before editing.
2. Identify the scoring mechanism and the smallest command that can reproduce the
   failure or verify the expected behavior.
3. Inspect existing files and task-local resources before making changes.
4. Make the smallest source change that satisfies the task and verifier contract.
5. Keep a compact record of the concrete evidence behind the change: observed
   failure, files inspected, edit made, and verifier result.
6. Run targeted verification before broad verification when practical.

## File Editing

1. Read the actual current file contents immediately before making any edit.
   Never rely on memory, prior snapshots, or assumed content.
2. Prefer direct in-place edits over patch or diff application when the exact
   current context is uncertain.
3. If using a patch or diff, confirm that every context line exists verbatim in
   the file before applying it.
4. If a patch hunk fails to apply, re-read the affected file region and perform
   the edit directly instead of retrying the same patch.
5. After any edit, re-read the affected region to confirm the change landed.

## Build and Test Fixes

When a task requires fixing a broken build, failing test, or generated artifact:

1. Run the relevant build, test, or verifier command first to capture the
   baseline failure.
2. Identify the specific error message, file, line, or expected output before
   editing.
3. Apply the smallest fix, then re-run the same targeted command.
4. Treat newly introduced failures as separate sub-tasks and resolve them in
   order.
5. Do not mark the task complete until the verifier-relevant command succeeds or
   the remaining failure is clearly outside the task boundary.

## Artifact-Contract Handling

Do not treat artifacts as ordinary text files. Treat them as contract-bearing
interfaces between input data, generated output, verifier checks, and downstream
consumers.

When a task requires reading, modifying, or generating an artifact such as JSON,
DOT, reports, configs, generated source, schemas, datasets, or parsed outputs:

1. Identify the artifact contract first: format, schema, required fields,
   identifiers, references, ordering, examples, verifier assertions, and
   consuming code.
2. Inspect representative source artifacts directly before deciding how to
   transform or preserve them.
3. Determine whether the task calls for preservation, transformation, repair,
   generation, or validation.
4. Preserve required literals, identifiers, references, ordering, and
   representative content unless the contract explicitly requires a change.
5. Do not invent, drop, rename, normalize, collapse, expand, or repair artifact
   elements unless the verifier or consumer contract requires that behavior.
6. Prefer structured parsers, serializers, validators, or existing consumer code
   over ad hoc string manipulation when they are available.
7. After producing the artifact, run targeted checks for parseability, required
   keys or IDs, reference consistency, expected counts, preserved content, and
   format-specific validity.
8. If targeted checks regress or become unusable after a change, stop expanding
   the solution. Re-inspect the source contract and narrow the edit before trying
   a broader repair.

A plausible-looking artifact is not sufficient evidence. The artifact is only
correct when it satisfies the task contract under the verifier or consuming
code.

## Constraints

- Do not bypass, remove, or weaken tests, verifier scripts, fixtures, or expected
  output checks.
- Do not treat this policy as overriding task-specific instructions or verifier
  requirements.
- On tool or environment errors, retry once when the retry is safe, then report
  the failure with the command and error output.
- On ambiguous instructions, make a conservative assumption and continue.

# Task Resources

Inspect the task files, environment, tests, and expected outputs directly.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=catpaw@example.com, author_name=CatPaw Task Engineer, category=document-editing, difficulty=medium, tags=[hwpx, xml-editing, document-processing, latent-method-reuse].
Verifier config: timeout_sec=600.0.