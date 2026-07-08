# Task Instruction

## Task: Fill HWPX Event Announcement Template

You need to replace all `{{...}}` placeholders in a `.hwpx` template with values from a JSON data file, then save the result as a valid `.hwpx` package.

### Step-by-step Plan

**Step 1: Inspect the workspace**
```bash
ls /root/
find /root/ -name 'event_data.json' -o -name 'event_announcement_template.hwpx' 2>/dev/null
```
Locate both `event_data.json` and `event_announcement_template.hwpx`. They may be in `/root/` or a subdirectory.

**Step 2: Read the JSON data file**
```bash
cat <path_to>/event_data.json
```
Note every key-value pair. These are the substitution values.

**Step 3: Understand the HWPX structure**
A `.hwpx` file is a ZIP archive containing XML files. Unzip it to a temp directory:
```bash
mkdir -p /tmp/hwpx_work
cp <path_to>/event_announcement_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_contents
find template_contents -type f -name '*.xml' | sort
```

**Step 4: Find all placeholders across ALL XML files**
```bash
grep -rn '{{\|}}' template_contents/
```
This will show every file containing placeholder markers. CRITICAL: Placeholders may be split across multiple XML tags (e.g., `<hp:t>{{</hp:t><hp:t>key}}</hp:t>`). You must handle this.

**Step 5: Write a Python script to perform the replacement**

Create `/tmp/hwpx_work/fill_template.py` with this approach:

1. For each XML file in the unzipped contents:
   a. Read the raw XML text.
   b. First, normalize split placeholders: collapse adjacent `<hp:t>` elements so that `{{key}}` patterns that are split across tags become single `<hp:t>{{key}}</hp:t>` elements. A reliable way: use regex to find sequences of `</hp:t></hp:run>...<hp:run>...<hp:t>` (or just `</hp:t><hp:t>` depending on actual structure) within what should be a single placeholder, and merge them. Alternatively, extract all text from consecutive `<hp:t>` nodes, concatenate, do replacement, then put back.
   c. After normalization, replace every `{{key}}` with the corresponding value from `event_data.json`.
   d. **Remove layout cache elements**: Strip all `<hp:lineSegArray>...</hp:lineSegArray>` and `<hp:lineSeg .../>` elements (and `<hp:lineSeg>...</hp:lineSeg>` variants) from any paragraph whose text was modified. To be safe, remove ALL `lineSegArray` and `lineSeg` elements from the entire document — this is the proven approach from prior successful HWPX tasks.
   e. Write the modified XML back.

2. Re-zip the modified contents into `/root/event_announcement_ready.hwpx`, preserving the original directory structure and using `zipfile` in Python (not command-line zip, to ensure proper structure).

Key implementation details for the Python script:
- Use `json.load()` to read event_data.json.
- For placeholder normalization: Read each XML file as a string. Use a regex approach to merge split `<hp:t>` tags. Pattern: `re.sub(r'</hp:t>(.*?)<hp:t>', lambda m: m.group(1) if not re.search(r'[^\s<>]', ... ) else m.group(0), text)` — but be careful. A simpler proven approach:
  - Extract text-only content between `<hp:t>` and `</hp:t>` tags in sequence, concatenate, check for `{{...}}` patterns, and if found, merge those tags.
  - Or: repeatedly apply `re.sub(r'(\{\{[^}]*)</hp:t>\s*(?:</?[^>]+>\s*)*<hp:t>([^}]*\}\})', r'\1\2', text)` until stable.
- After merging, do simple string replacement for each `{{key}}` → value.
- Remove lineSeg/lineSegArray: `re.sub(r'<hp:lineSegArray[^>]*>.*?</hp:lineSegArray>', '', text, flags=re.DOTALL)` and similarly for `<hp:lineSeg[^>]*/?>`.
- Verify no `{{` remains in any XML file after processing.

**Step 6: Run the script**
```bash
cd /tmp/hwpx_work && python3 fill_template.py
```

**Step 7: Validate the output**
```bash
# Check it's a valid zip
python3 -c "import zipfile; z=zipfile.ZipFile('/root/event_announcement_ready.hwpx'); z.testzip(); print('Valid ZIP'); z.close()"

# Check no placeholders remain
mkdir -p /tmp/hwpx_verify
cd /tmp/hwpx_verify && unzip /root/event_announcement_ready.hwpx -d verify_contents
grep -rn '{{' verify_contents/ && echo 'FAIL: placeholders remain' || echo 'PASS: no placeholders'

# Check Korean labels are preserved (spot check a few)
grep -r '행사' verify_contents/ | head -5

# Check lineSegArray is removed
grep -rn 'lineSegArray\|lineSeg' verify_contents/ && echo 'WARNING: lineSeg elements remain' || echo 'PASS: lineSeg removed'
```

**Step 8: Run the verifier if available**
```bash
find /root/ -name 'test_output.py' -o -name 'verify*' 2>/dev/null
# If found, run it:
cd <task_dir> && python3 -m pytest test_output.py -v
```

### Critical Reminders
- Placeholders WILL be split across XML tags. You MUST handle this. Do not assume `{{key}}` appears as a single text run.
- ALL `lineSegArray` and `lineSeg` elements must be removed from modified sections.
- The output path must be exactly `/root/event_announcement_ready.hwpx`.
- Korean text and static note lines must be preserved exactly.
- The result must be a valid ZIP (`.hwpx` is a ZIP-based format).

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