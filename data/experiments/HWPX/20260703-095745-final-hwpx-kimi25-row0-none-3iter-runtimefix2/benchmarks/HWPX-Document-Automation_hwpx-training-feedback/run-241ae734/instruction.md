# Task Instruction

Complete the following task to fill in a training feedback HWPX template.

## Goal
Fill in `training_feedback_template.hwpx` using values from `training_feedback.json`, save the result to `/root/training_feedback_ready.hwpx`.

## Steps

### 1. Inspect the input files
- Read `training_feedback.json` to understand all available fields and their values.
- Extract the HWPX template (it's a ZIP file) and inspect all XML files inside, especially any `section*.xml` files under `Contents/`. Identify every `{{...}}` placeholder and which JSON key it maps to.
- Note the exact XML namespace prefixes used (e.g., `hp:p`, `hp:t`, `hp:lineSegArray` or `hp:linesegarray` — check the actual case used in the file).

### 2. Prepare transformed values (do this BEFORE any XML manipulation)
- **참석자수**: Extract only the digits from the JSON value (e.g., '25명' → '25'). Use regex `re.sub(r'[^0-9]', '', value)`.
- **만족도**: Format as `{score}점 (5.0점 만점)` where `{score}` is the numeric score from the JSON (e.g., if JSON has '4.5' or '4.5/5.0', extract the score and format as '4.5점 (5.0점 만점)').
- **Overall opinion / 종합의견**: Take the comment value from JSON and append ` 후속 심화반 검토 요망.` at the end (with a space before it if the original doesn't end with a space).
- All other placeholders: use the JSON values as-is.

### 3. Write a Python script to perform the replacement
Use the following proven pattern for HWPX manipulation:

```python
import zipfile, json, re, os, shutil, copy
from lxml import etree

# Load JSON
with open('training_feedback.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare replacement map: placeholder_key -> replacement_value
# Build this map from the JSON keys, applying the transformations above
# e.g., replacements = {'교육명': data['교육명'], '참석자수': re.sub(...), ...}

# Process each XML file in the HWPX
# For each <hp:p> element:
#   1. Collect all <hp:t> text nodes and concatenate them
#   2. Check if the concatenated text contains any {{...}} placeholder
#   3. If yes:
#      a. Replace ALL placeholders in the concatenated text with their values
#      b. Clear all <hp:t> nodes except the first one
#      c. Set the first <hp:t> node's text to the fully replaced string
#      d. Remove any <hp:lineSegArray> (or <hp:linesegarray>) child element
#         from this <hp:p> — THIS IS CRITICAL
#   4. If no placeholder but the paragraph text contains any replacement value
#      that was just injected (check this as a safety measure), also remove
#      lineSegArray
```

### 4. Critical details for the XML processing
- **Tag fragmentation**: Placeholders like `{{교육명}}` may be split across multiple `<hp:t>` tags (e.g., `{{`, `교육명`, `}}`). Always concatenate ALL `<hp:t>` text within a `<hp:p>` before checking for placeholders.
- **Namespace handling**: Parse the XML namespace map from the root element. Use it consistently when finding elements. The namespace for `hp` is typically `http://www.hancom.co.kr/hwpml/2011/paragraph` or similar — read it from the actual file.
- **lineSegArray removal**: After modifying any paragraph's text, find and remove ALL `lineSegArray` elements (regardless of namespace prefix case) that are children of that `<hp:p>`. Use the namespace-aware search. This prevents overlapping character rendering.
- **Verify no remaining placeholders**: After all replacements, scan the entire XML for any remaining `{{` patterns. If found, flag them and fix.

### 5. Repackage the HWPX
- Create the output ZIP file at `/root/training_feedback_ready.hwpx`.
- Copy ALL files from the original HWPX archive into the new one.
- For XML files that were modified, write the modified version; for all others, copy the original bytes.
- Use `ZIP_DEFLATED` compression.
- Ensure directory structure is preserved exactly (including `[Content_Types].xml`, `META-INF/`, `Contents/`, etc.).

### 6. Validate the output
- Open the output file as a ZIP and verify it's valid.
- Read back the modified XML files and confirm:
  - No `{{` or `}}` patterns remain anywhere.
  - 참석자수 value is digits only (no Korean characters).
  - 만족도 follows the `X.X점 (5.0점 만점)` format.
  - The overall opinion ends with `후속 심화반 검토 요망.`
  - No `lineSegArray` elements exist in any paragraph that contains replacement text.
  - All Korean labels and static note lines are unchanged.
- Print confirmation of each validation check.

## Important Reminders
- Do NOT skip the lineSegArray removal step — a previous similar task failed specifically because of this.
- Do NOT modify paragraphs that don't contain placeholders (except for lineSegArray cleanup if their text was changed).
- The file MUST be saved to exactly `/root/training_feedback_ready.hwpx`.

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