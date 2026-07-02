# Task Instruction

Complete the following task to prepare an event announcement HWPX document.

## Goal
Replace all `{{...}}` placeholders in `event_announcement_template.hwpx` with values from `event_data.json`, and save the result to `/root/event_announcement_ready.hwpx`.

## Step-by-step Plan

### Step 1: Examine the JSON data file
Read `event_data.json` to understand the keys and values available for substitution.
```
cat event_data.json
```

### Step 2: Understand the HWPX structure
HWPX files are ZIP archives. List the contents:
```
python3 -c "import zipfile; z=zipfile.ZipFile('event_announcement_template.hwpx','r'); print('\n'.join(z.namelist())); z.close()"
```

### Step 3: Inspect XML files for placeholders
Extract and inspect each XML file inside the HWPX to find where `{{...}}` placeholders appear. Focus especially on files in `Contents/` or `section*.xml` paths. Print the content of each XML file:
```python
import zipfile
z = zipfile.ZipFile('event_announcement_template.hwpx', 'r')
for name in z.namelist():
    try:
        content = z.read(name).decode('utf-8')
        if '{{' in content:
            print(f'=== {name} ===')
            print(content)
    except:
        pass
z.close()
```

### Step 4: Check for split placeholders
Placeholders like `{{event_name}}` might be split across multiple XML runs/elements (e.g., `{{`, `event_name`, `}}`). You MUST handle this. After reading each XML file's text, check if placeholders are intact or fragmented across XML tags.

### Step 5: Write the replacement script
Write a Python script that:
1. Opens the template HWPX as a ZIP archive.
2. Reads `event_data.json` to get the replacement mapping.
3. For each file in the ZIP:
   a. If it's an XML file containing `{{` placeholders (after handling potential splits across XML elements):
      - For text-bearing elements (likely `<hp:t>` or `<t>` tags), concatenate adjacent text runs within the same paragraph to detect and replace complete `{{key}}` patterns.
      - **Critical**: Handle the case where a single `{{key}}` placeholder is split across multiple `<hp:t>` elements or text runs within the same paragraph. The approach should be:
        - For each paragraph element, collect all text content, perform the replacement on the combined text, then redistribute back OR merge the runs.
        - Alternatively, do a regex-based approach on the raw XML that can match `\{\{[^}]+\}\}` even across XML tags, but be very careful with this approach.
      - Replace each `{{key}}` with the corresponding value from the JSON.
      - **Remove layout-cache elements** from any paragraph whose text was modified. These are typically elements like `<hp:linesegarray>...</hp:linesegarray>` or `<linesegarray>...</linesegarray>` or similar caching elements. Use regex or XML parsing to remove them from modified paragraphs.
   b. If not an XML file with placeholders, copy it as-is.
4. Write the result to `/root/event_announcement_ready.hwpx` as a valid ZIP.
5. Preserve the original ZIP compression method and structure.

### Step 6: Validate the output
After creating the output file:
1. Verify it's a valid ZIP:
   ```
   python3 -c "import zipfile; z=zipfile.ZipFile('/root/event_announcement_ready.hwpx','r'); print('Valid ZIP, files:', z.namelist()); z.close()"
   ```
2. Verify NO `{{...}}` placeholders remain:
   ```python
   import zipfile, re
   z = zipfile.ZipFile('/root/event_announcement_ready.hwpx', 'r')
   for name in z.namelist():
       try:
           content = z.read(name).decode('utf-8')
           matches = re.findall(r'\{\{.*?\}\}', content)
           if matches:
               print(f'REMAINING PLACEHOLDERS in {name}: {matches}')
       except: pass
   print('Validation complete')
   z.close()
   ```
3. Verify Korean labels and static content are preserved by printing the text content of section XML files.
4. Verify that layout-cache elements (`linesegarray` or similar) have been removed from modified paragraphs.

## Important Notes
- Do NOT use `shutil.copy` of the template as the output without processing.
- Preserve all non-placeholder content exactly, including Korean text.
- The HWPX namespace prefixes (like `hp:`, `hc:`, etc.) must be preserved.
- When removing layout-cache elements, only remove them from paragraphs where text was actually modified, not from all paragraphs.
- If placeholders are NOT split across elements, simple string replacement on the XML content is fine. Only use the complex merging approach if splits are detected.
- Make sure the output ZIP preserves the directory structure and all original files.

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