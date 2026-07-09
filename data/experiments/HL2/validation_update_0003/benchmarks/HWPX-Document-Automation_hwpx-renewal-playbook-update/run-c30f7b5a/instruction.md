# Task Instruction

## Task: Update renewal_playbook.hwpx with new data and save to /root/renewal_playbook_updated.hwpx

### Step 0 — Explore the workspace and understand all inputs

```bash
cd /root
find . -maxdepth 2 -type f | head -60
```

Locate:
- `renewal_playbook.hwpx` (the source HWPX file)
- `renewal_update.json` (field updates)
- `followups.csv` (follow-up lines)
- `test_outputs.py` (verifier — read but do NOT modify)

### Step 1 — Read the update data

```bash
cat renewal_update.json
cat followups.csv
```

Note every key-value pair in the JSON (customer name, current owner, renewal window, pricing band, escalation contact, pricing note, and any others). Note the CSV rows and their `sequence` column for ordering.

### Step 2 — Understand the HWPX package structure

An `.hwpx` file is a ZIP archive. Unzip it to inspect:

```bash
mkdir -p /tmp/hwpx_src
cp renewal_playbook.hwpx /tmp/hwpx_src/source.hwpx
cd /tmp/hwpx_src
unzip -o source.hwpx -d extracted
find extracted -type f
```

Identify all XML files, especially `Contents/section0.xml`, `Contents/section1.xml`, etc. Read each section XML file completely:

```bash
cat extracted/Contents/section0.xml
cat extracted/Contents/section1.xml
```

Also check for any other section files or content files.

### Step 3 — Read the verifier to understand the contract

```bash
cat /root/test_outputs.py
```

Pay close attention to:
- Which values must appear in section0_xml or section1_xml
- Which old values must NOT appear
- The follow-up replacement check
- The appendix preservation check (`이 부록 문단은 그대로 유지해야 합니다.`)
- How the verifier extracts section XML (it likely unzips and reads specific paths)
- Any checks on HWPX validity (mimetype, zip structure, etc.)

### Step 4 — Identify old values and plan replacements

From the JSON, build a mapping of old→new. The old values will be visible in the section XML files you read in Step 2. For each field in the JSON update:
- Search the XML text content for the old value
- Plan the exact string replacement

For follow-ups:
- Identify the three existing follow-up lines in the XML
- Plan to replace them with the CSV items ordered by `sequence`

### Step 5 — Write a Python script to perform the update

Create `/tmp/do_update.py` that:

1. Reads `renewal_update.json` and `followups.csv`.
2. Unzips the source HWPX into a temp directory.
3. For each section XML file (`section0.xml`, `section1.xml`, etc.):
   a. Parses the XML properly using `lxml.etree` (or `xml.etree.ElementTree`).
   b. Walks ALL text-bearing elements. In HWPX/HWPML, text is typically inside `<hp:t>` tags or similar. Iterate over ALL elements and check `.text` and `.tail` attributes.
   c. For each text node, performs string replacements for ALL fields from the JSON.
   d. Replaces follow-up lines according to the CSV (match old follow-up text, replace with new in sequence order).
   e. **Critical**: After modifying any `<hp:t>` element's text, remove any sibling or child elements that serve as layout cache (e.g., `<hp:linesegarray>`, `<hp:lineSegArray>`, `<lineseg>`, or similar layout/cache elements within the parent paragraph `<hp:p>`). Search for these by inspecting the XML structure. If a `<hp:p>` paragraph contains a modified run, remove layout cache sub-elements from that paragraph.
   f. Preserves the appendix sentence exactly.
   g. Serializes the XML back with the same encoding and XML declaration.
4. Re-packs the HWPX as a ZIP file:
   - The `mimetype` file (if present) must be stored first, uncompressed (store method).
   - All other files use deflate compression.
   - Preserves the exact directory structure.
5. Saves to `/root/renewal_playbook_updated.hwpx`.

**Important implementation details:**
- Use namespace-aware XML parsing. Inspect the XML to find the exact namespace URIs and tag names.
- When searching for text to replace, check EVERY element's `.text` and `.tail` in the entire XML tree, not just specific tag names.
- For follow-up replacement: identify the old follow-up lines from the original XML, then replace them with CSV items sorted by `sequence`.
- After ALL replacements, verify by re-reading the output XML that old values are gone and new values are present.

### Step 6 — Run the update script

```bash
cd /root
python3 /tmp/do_update.py
```

Check for any errors.

### Step 7 — Validate the output before running the verifier

```bash
# Verify it's a valid zip
unzip -t /root/renewal_playbook_updated.hwpx

# Extract and check the section XMLs contain new values
mkdir -p /tmp/verify
cd /tmp/verify
unzip -o /root/renewal_playbook_updated.hwpx -d check
cat check/Contents/section0.xml | grep -o 'Asteron Commerce' | head -3
# Check for old values that should be gone
cat check/Contents/section0.xml
cat check/Contents/section1.xml
```

Verify:
- New customer name appears
- Old customer name is gone
- All other updated fields appear and old versions are gone
- Follow-up lines are replaced
- Appendix sentence is preserved
- No stale layout cache in modified paragraphs

### Step 8 — Run the verifier

```bash
cd /root
python3 -m pytest test_outputs.py -v
```

If any test fails, read the error carefully, inspect the relevant XML, fix the update script, and re-run.

### Key Pitfalls to Avoid (from prior failures)
- The prior run failed because updated values were not found in the section XML. This likely means the replacement logic didn't target the right XML elements or didn't save correctly. Make sure to:
  1. Actually inspect the raw XML to see where text lives (it may be in `<hp:t>` tags with specific namespaces)
  2. Walk the ENTIRE element tree, not just top-level elements
  3. Properly serialize back to the same file paths in the ZIP
  4. Use the correct namespace prefixes when writing back
- Do NOT use `str.replace()` on raw XML strings for content replacement — use proper XML parsing to avoid breaking tags. However, if the XML parser approach proves difficult, a careful text-level replacement on the serialized XML (being careful not to touch tag names/attributes) is acceptable as a fallback.
- Make sure the ZIP repacking preserves the exact structure of the original HWPX.

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