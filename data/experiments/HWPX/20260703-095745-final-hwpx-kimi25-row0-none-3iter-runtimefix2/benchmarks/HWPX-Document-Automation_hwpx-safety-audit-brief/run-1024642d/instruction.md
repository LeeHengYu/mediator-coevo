# Task Instruction

Complete the following steps to prepare the warehouse safety audit brief:

## Step 1: Inspect available files
- List files in the working directory and any subdirectories to find `safety_audit_template.hwpx`, `audit_overview.json`, and `corrective_actions.json`.
- Read both JSON files completely and note all field names and values.
- Note the risk tier value and map it: High -> 즉시조치, Medium -> 계획보완, Low -> 모니터링.
- Note the inspection date and prepare the YYYY.MM.DD version (replace hyphens with dots).

## Step 2: Unpack the HWPX template
- Copy `safety_audit_template.hwpx` to a working location.
- Unzip it into a temporary directory (e.g., `/root/hwpx_work/`): `mkdir -p /root/hwpx_work && cd /root/hwpx_work && unzip <path_to_template>`.
- List all files in the extracted archive to understand the package structure.
- Identify the main content XML files (likely under `Contents/` — look for files like `section0.xml` or `content.hpf` or similar).

## Step 3: Examine the template XML content
- Read each content XML file carefully. Look for:
  - `{{...}}` placeholder patterns — list ALL of them.
  - Section titles and row labels (these must be preserved exactly).
  - The structure of the summary/overview section.
  - The structure of the audit table with value cells.
  - The three corrective-action lines.
  - Any occurrences of risk tier placeholders.
  - Any occurrences of date placeholders.
- Also look for layout-cache elements (often named something like `lineseg`, `lineSegArray`, `LineSeg`, or similar cache/layout elements within paragraph tags).

## Step 4: Plan and execute replacements
Using Python for precise XML manipulation:

```python
import json, os, shutil, zipfile
from pathlib import Path
import re

# Read JSON data
# ... load audit_overview.json and corrective_actions.json

# For each content XML file in the unpacked HWPX:
# 1. Read the XML as text (to handle namespaces gracefully)
# 2. Replace ALL {{placeholder}} patterns with corresponding values from the JSON files
# 3. For the risk tier: replace every occurrence with the tier value followed by a space and the Korean severity note
# 4. For dates: convert every YYYY-MM-DD date to YYYY.MM.DD format
# 5. For corrective actions: fill the three lines in the SAME ORDER as corrective_actions.json
# 6. Remove layout-cache elements (e.g., <hp:lineseg ...>...</hp:lineseg> or similar) from ANY paragraph whose text content was modified
# 7. Verify no {{...}} placeholders remain anywhere in the file
```

## Step 5: Handle layout-cache removal carefully
- When modifying paragraph text, identify the parent paragraph element.
- Remove any child elements that represent layout caches (common names: `lineseg`, `lineSegArray`, `LineSeg`, `lineBreak` cache elements). These are typically elements that store pre-computed glyph positions.
- Use XML parsing (lxml or ElementTree) for this step to ensure structural validity. Be careful with namespaces.

## Step 6: Repack the HWPX
- Rezip the modified directory into `/root/safety_audit_brief_final.hwpx`.
- IMPORTANT: Use `zipfile.ZipFile` with `ZIP_DEFLATED` compression. Preserve the original directory structure exactly. The `mimetype` file (if present) should be stored first without compression (ZIP_STORED), similar to ODF/EPUB conventions.
- Verify the ZIP is valid: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/safety_audit_brief_final.hwpx'); z.testzip(); print('Valid ZIP')"`

## Step 7: Final validation
- Unzip the final HWPX to a verification directory.
- Search ALL XML files for any remaining `{{` or `}}` patterns — there must be NONE.
- Confirm the date appears in YYYY.MM.DD format (not YYYY-MM-DD) everywhere.
- Confirm the risk tier has the severity note appended everywhere it appears.
- Confirm corrective actions appear in the correct order.
- Confirm section titles and row labels are unchanged.
- Print a summary of all replacements made.

## Critical rules:
- Do NOT change section titles or row labels.
- Do NOT leave any `{{...}}` placeholder text.
- The severity note must appear IMMEDIATELY AFTER the risk tier text (e.g., 'High 즉시조치').
- Corrective actions must be in the SAME ORDER as in corrective_actions.json.
- Every date must use dots not hyphens.
- Every modified paragraph must have its layout-cache sub-elements removed.
- The output must be a valid .hwpx ZIP package at `/root/safety_audit_brief_final.hwpx`.

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