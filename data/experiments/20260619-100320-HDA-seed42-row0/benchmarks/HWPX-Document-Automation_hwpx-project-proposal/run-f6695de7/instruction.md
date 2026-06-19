# Task Instruction

Execute the following steps to produce `/root/project_proposal_ready.hwpx`.

## Step 0 – Inspect the inputs

```bash
cd /root
ls -la
cat project_proposal.json
```

Unzip the template to a temp directory and list the contents:

```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
rm -rf *
unzip /root/project_proposal_template.hwpx -d template
find template -type f
```

Read every XML section file (especially `Contents/section*.xml`) carefully:

```bash
for f in template/Contents/section*.xml; do echo "=== $f ==="; cat "$f"; echo; done
```

Also check `mimetype` and `META-INF/` contents.

## Step 1 – Write the Python script

Create `/tmp/hwpx_work/build.py` with the following logic:

```python
import json, os, re, shutil, zipfile
import xml.etree.ElementTree as ET

# ---- paths ----
TEMPLATE = '/root/project_proposal_template.hwpx'
JSON_FILE = '/root/project_proposal.json'
OUTPUT   = '/root/project_proposal_ready.hwpx'
WORK     = '/tmp/hwpx_work/template'

# ---- load JSON values ----
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Flatten nested JSON into a flat dict keyed by placeholder name.
# The JSON may be flat or nested; handle both.
def flatten(obj, prefix=''):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                out.update(flatten(v, k + '.'))
            else:
                out[k] = v
                if prefix:
                    out[prefix + k] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, prefix))
    return out

flat = flatten(data)
print("Flat JSON keys:", list(flat.keys()))

# ---- process each section XML ----
NS = {'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph',
      'hp_char': 'http://www.hancom.co.kr/hwpml/2011/character'}

# Register all namespaces found in the file so they are preserved on write.
# We will use string-based processing for reliability.

for secfile in sorted([os.path.join(WORK, 'Contents', f)
                       for f in os.listdir(os.path.join(WORK, 'Contents'))
                       if f.endswith('.xml')]):
    print(f'Processing {secfile}')
    with open(secfile, 'r', encoding='utf-8') as f:
        xml = f.read()

    # --- A. Consolidate split placeholder tags ---
    # HWPX often splits {{placeholder}} across multiple <hp:t> tags inside
    # the same <hp:run>. We consolidate text within each <hp:run>.
    # Strategy: find each <hp:run>...</hp:run>, merge all <hp:t> texts,
    # then put the merged text in a single <hp:t> and remove extras.

    def merge_run_texts(run_match):
        run_xml = run_match.group(0)
        # Extract all <hp:t ...>TEXT</hp:t> contents
        t_pattern = re.compile(r'<hp:t[^>]*>(.*?)</hp:t>', re.DOTALL)
        texts = t_pattern.findall(run_xml)
        if len(texts) <= 1:
            return run_xml
        merged = ''.join(texts)
        # Replace: keep first <hp:t>, put merged text, remove rest
        first = True
        def replacer(m):
            nonlocal first
            if first:
                first = False
                return m.group(0).replace(m.group(1), merged) if m.group(1) != merged else f'<hp:t>{merged}</hp:t>'
            return ''  # remove subsequent <hp:t> tags
        run_xml_new = t_pattern.sub(replacer, run_xml)
        return run_xml_new

    xml = re.sub(r'<hp:run[^>]*>.*?</hp:run>', merge_run_texts, xml, flags=re.DOTALL)

    # --- B. Replace {{...}} placeholders with JSON values ---
    def replace_placeholder(m):
        key = m.group(1).strip()
        if key in flat:
            val = str(flat[key])
            # Normalize budget: remove commas but keep currency symbol
            # Detect if value looks like currency with commas
            if re.match(r'^[^0-9]*[\d,]+$', val) and ',' in val:
                val = val.replace(',', '')
            return val
        # Try partial key matching (last segment)
        for k, v in flat.items():
            if k.endswith(key) or key.endswith(k):
                val = str(v)
                if re.match(r'^[^0-9]*[\d,]+$', val) and ',' in val:
                    val = val.replace(',', '')
                return val
        print(f'  WARNING: placeholder {{{{{key}}}}} not found in JSON')
        return m.group(0)  # leave as-is if not found

    xml = re.sub(r'\{\{\s*(.*?)\s*\}\}', replace_placeholder, xml)

    # --- C. Append month span after phase (단계) lines ---
    # Find lines containing 단계1, 단계2, 단계3 etc. with date ranges,
    # and append (N개월) based on the date range.
    # Date formats may be YYYY-MM or YYYY.MM; handle both.
    def add_month_span(m):
        full = m.group(0)
        # Already has month span appended?
        if '개월)' in full:
            return full
        # Find date range pattern: YYYY-MM ~ YYYY-MM or YYYY.MM ~ YYYY.MM
        date_match = re.search(r'(\d{4})[-.](\d{2})\s*~\s*(\d{4})[-.](\d{2})', full)
        if date_match:
            y1, m1, y2, m2 = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)), int(date_match.group(4))
            months = (y2 - y1) * 12 + (m2 - m1) + 1
            span_text = f' ({months}개월)'
            # Insert before </hp:t>
            # Find the last </hp:t> in this block and insert before it
            idx = full.rfind('</hp:t>')
            if idx >= 0:
                return full[:idx] + span_text + full[idx:]
            else:
                return full + span_text
        return full

    # Apply to runs/paragraphs containing 단계
    # We need to match the <hp:run> (or broader context) that contains 단계 and a date range
    xml = re.sub(r'<hp:run[^>]*>(?:(?!</hp:run>).)*단계\d(?:(?!</hp:run>).)*</hp:run>', add_month_span, xml, flags=re.DOTALL)

    # If 단계 text and date range are in different runs within the same paragraph,
    # we need a paragraph-level approach:
    def add_month_span_para(m):
        para_xml = m.group(0)
        # Check if this paragraph has 단계 and a date range
        if '단계' not in para_xml:
            return para_xml
        date_match = re.search(r'(\d{4})[-.](\d{2})\s*~\s*(\d{4})[-.](\d{2})', para_xml)
        if not date_match:
            return para_xml
        if '개월)' in para_xml:
            return para_xml
        y1, m1, y2, m2 = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)), int(date_match.group(4))
        months = (y2 - y1) * 12 + (m2 - m1) + 1
        span_text = f' ({months}개월)'
        # Find the last </hp:t> before the closing of the paragraph and insert
        idx = para_xml.rfind('</hp:t>')
        if idx >= 0:
            return para_xml[:idx] + span_text + para_xml[idx:]
        return para_xml

    xml = re.sub(r'<hp:p\b[^>]*>.*?</hp:p>', add_month_span_para, xml, flags=re.DOTALL)

    # --- D. Remove ALL <hp:lineSegArray> elements (layout cache) ---
    # Use a robust pattern that handles namespace prefixes and attributes
    xml = re.sub(r'<hp:lineSegArray[^>]*>.*?</hp:lineSegArray>', '', xml, flags=re.DOTALL)
    # Also handle self-closing variant
    xml = re.sub(r'<hp:lineSegArray[^/]*/>', '', xml)
    # Handle possible namespace variations (e.g., without prefix, or different prefix)
    xml = re.sub(r'<[^>]*lineSegArray[^>]*>.*?</[^>]*lineSegArray>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'<[^>]*lineSegArray[^/]*/>', '', xml)

    # --- E. Verify no remaining placeholders ---
    remaining = re.findall(r'\{\{.*?\}\}', xml)
    if remaining:
        print(f'  WARNING: remaining placeholders: {remaining}')

    with open(secfile, 'w', encoding='utf-8') as f:
        f.write(xml)

# ---- repackage as HWPX ----
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

with zipfile.ZipFile(OUTPUT, 'w') as zf:
    # mimetype must be first and stored uncompressed
    mimetype_path = os.path.join(WORK, 'mimetype')
    if os.path.exists(mimetype_path):
        zf.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)

    for root, dirs, files in os.walk(WORK):
        for fname in sorted(files):
            full = os.path.join(root, fname)
            arcname = os.path.relpath(full, WORK)
            if arcname == 'mimetype':
                continue
            zf.write(full, arcname, compress_type=zipfile.ZIP_DEFLATED)

print('Output written to', OUTPUT)
```

## Step 2 – Run the script

```bash
cd /tmp/hwpx_work
python3 build.py
```

## Step 3 – Validate the output

```bash
# Check it's a valid zip
python3 -c "import zipfile; z=zipfile.ZipFile('/root/project_proposal_ready.hwpx'); print(z.namelist()[:20])"

# Check no remaining placeholders
mkdir -p /tmp/hwpx_verify
cd /tmp/hwpx_verify
rm -rf *
unzip /root/project_proposal_ready.hwpx -d verify
for f in verify/Contents/section*.xml; do
  echo "=== $f ==="
  # Check for remaining placeholders
  if grep -o '{{[^}]*}}' "$f"; then
    echo "ERROR: remaining placeholders found!"
  else
    echo "OK: no placeholders"
  fi
  # Check for 개월
  grep -o '[0-9]*개월)' "$f" && echo "Month spans found" || echo "WARNING: no month spans"
  # Check for lineSegArray
  if grep -i 'lineSegArray' "$f"; then
    echo "ERROR: lineSegArray still present!"
  else
    echo "OK: no lineSegArray"
  fi
  # Show phase lines for verification
  grep -o '단계[0-9][^<]*' "$f" || true
done
```

## Step 4 – If any validation fails

- Re-read the section XML files from the verify directory to understand the actual structure.
- Fix the script accordingly and re-run.
- Pay special attention to:
  - Date format in the actual XML (dashes vs dots)
  - Whether 단계 text and date ranges are in the same `<hp:run>` or split across runs
  - Budget value format and the placeholder key name
  - Any namespace variations for lineSegArray

## Key Points from Previous Feedback

1. **Date format**: The actual data uses dashes (`2026-08 ~ 2026-10`), NOT dots. The regex must handle `YYYY-MM` format. The script above handles both.
2. **Month span calculation**: Use `(y2-y1)*12 + (m2-m1) + 1` to get inclusive month count.
3. **lineSegArray removal**: Use XML-parser-level or very robust regex. The script removes ALL variants.
4. **Budget normalization**: Remove commas from numeric budget values while keeping the currency symbol (e.g., `₩` or `원`).
5. **HWPX packaging**: `mimetype` file must be first entry, stored uncompressed.
6. **Split placeholders**: HWPX often splits `{{placeholder}}` across multiple `<hp:t>` tags; consolidate within `<hp:run>` before replacement.

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