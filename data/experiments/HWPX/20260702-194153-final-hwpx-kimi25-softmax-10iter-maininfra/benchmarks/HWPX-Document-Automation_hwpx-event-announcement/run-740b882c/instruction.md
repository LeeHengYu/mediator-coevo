# Task Instruction

Complete the following task to prepare an event announcement HWPX document.

## Context

A `.hwpx` file is a ZIP archive containing XML files (Hancom/Hangul word processor format). The template `event_announcement_template.hwpx` contains `{{...}}` placeholders that must be replaced with values from `event_data.json`. The result must be saved to `/root/event_announcement_ready.hwpx`.

## Steps

### 1. Inspect the data file
```bash
cat /root/event_data.json
```
Note all key-value pairs. These keys correspond to placeholder names like `{{key_name}}`.

### 2. Explore the HWPX template structure
```bash
cd /root
mkdir -p hwpx_work
cp event_announcement_template.hwpx hwpx_work/template.zip
cd hwpx_work
unzip template.zip -d template_extracted
find template_extracted -type f | sort
```
List all files to understand the package structure.

### 3. Identify files containing placeholders
```bash
grep -rl '{{' template_extracted/
```
This will show which XML (or other) files contain `{{...}}` placeholders. Inspect each one:
```bash
for f in $(grep -rl '{{' template_extracted/); do echo "=== $f ==="; cat "$f"; echo; done
```

### 4. Write a Python script to perform replacements and clean layout caches

Create `/root/hwpx_work/process.py` that:

a) Reads `event_data.json` to get the replacement mapping.

b) For each file in the extracted template that contains `{{`:
   - Loads the file content as text.
   - **IMPORTANT**: Placeholders may be split across multiple XML elements/tags (e.g., `<hp:t>{{</hp:t><hp:t>event_name</hp:t><hp:t>}}</hp:t>`). Before doing simple text replacement, first try replacing on the raw XML string. If that doesn't catch all placeholders, parse the XML, concatenate text content of paragraph runs, detect placeholders spanning multiple text nodes, and handle accordingly.
   - Replaces each `{{key}}` with the corresponding value from the JSON.
   - For any paragraph (`<hp:p>` element) whose text content was modified, removes all `<hp:linesegarray>` elements (and their children) within that paragraph. These are layout-cache elements that store pre-computed glyph positions. Also remove any `<hp:lineSeg>` elements or similar cached layout data within modified paragraphs. Check for elements like `<linesegarray>`, `<LineSeg>`, `<hp:lineseg>` — inspect the actual XML namespace and element names used.
   - Writes the modified content back.

c) After all replacements, verifies no `{{` remains in any file in the extracted directory.

d) Preserves all Korean labels and static note lines (they should be untouched since we only replace `{{...}}` patterns).

### 5. Repackage as HWPX
```bash
cd /root/hwpx_work/template_extracted
zip -r -0 /root/event_announcement_ready.hwpx mimetype
zip -r /root/event_announcement_ready.hwpx . -x mimetype
```
Note: If there's a `mimetype` file, it should be stored first without compression (like ODF packages). If there's no `mimetype` file, just zip everything normally:
```bash
cd /root/hwpx_work/template_extracted
zip -r /root/event_announcement_ready.hwpx .
```

### 6. Validate the result
```bash
# Verify it's a valid ZIP
unzip -t /root/event_announcement_ready.hwpx

# Verify no placeholders remain
unzip -p /root/event_announcement_ready.hwpx | grep -c '{{'
# Should output 0

# More thorough check across all files
for f in $(unzip -l /root/event_announcement_ready.hwpx | awk '/-----/{p=1;next} p{print $4}' | grep -v '^$'); do
  content=$(unzip -p /root/event_announcement_ready.hwpx "$f" 2>/dev/null)
  if echo "$content" | grep -q '{{'; then
    echo "PLACEHOLDER FOUND IN: $f"
  fi
done
```

## Critical Details

- **Layout cache removal**: This is essential. When you modify text in a paragraph, any `<hp:linesegarray>` (or similarly named layout cache element) in that paragraph MUST be removed, otherwise the document will display with overlapping characters. Inspect the actual XML element names in the template before writing the removal logic.
- **Namespace handling**: HWPX XML files use namespaces. When parsing with Python's `xml.etree.ElementTree`, register namespaces properly to avoid mangling them on output. Use `ET.register_namespace()` for all namespaces found in the file. Alternatively, use string-based replacement if the XML structure is simple enough.
- **Encoding**: Ensure UTF-8 encoding is preserved throughout (Korean text).
- **Do NOT modify** any Korean labels or the static note line — only replace `{{...}}` patterns.
- The output path must be exactly `/root/event_announcement_ready.hwpx`.

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