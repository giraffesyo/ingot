---
name: QwenImage UI
description: A compact local image workbench with visible queue and execution state.
colors:
  page: "#f7f8f7"
  surface: "#fff"
  canvas: "#edf0ed"
  ink: "#24302a"
  muted: "#5c6860"
  line: "#d6ddd7"
  accent: "#246447"
  accent-hover: "#174c34"
  error: "#9b302a"
  field-border: "#b5c0b7"
  queue-selected: "#dfece3"
  queue-hover: "#e8ede9"
typography:
  body:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  title:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "14px"
    fontWeight: 600
    lineHeight: 1.5
  label:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "13px"
    fontWeight: 600
    lineHeight: 1.5
  hint:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.55
  log:
    fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.7
rounded:
  radius: "6px"
  canvas: "8px"
  thumbnail: "4px"
spacing:
  "8": "8px"
  "14": "14px"
  "18": "18px"
  "22": "22px"
  "32": "32px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.surface}"
    rounded: "{rounded.radius}"
    padding: "8px 14px"
  button-primary-hover:
    backgroundColor: "{colors.accent-hover}"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.radius}"
    padding: "8px 14px"
  button-text:
    backgroundColor: "transparent"
    textColor: "{colors.muted}"
    rounded: "{rounded.radius}"
    padding: "4px"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.radius}"
    padding: "9px 11px"
  preview:
    backgroundColor: "{colors.canvas}"
    rounded: "{rounded.canvas}"
  queue-row:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "0"
    padding: "11px 10px"
  queue-row-selected:
    backgroundColor: "{colors.queue-selected}"
  log:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
    typography: "{typography.log}"
    rounded: "{rounded.radius}"
    padding: "14px"
---

# Design System: QwenImage UI

## Overview

**Creative North Star: "The local image workbench"**

A quiet, compact utility keeps controls legible and lets generated images occupy the largest surface. Light graphite, white fields, forest green actions, and system typography provide the visual hierarchy.

This records the standalone module's implemented system in `web/style.css`, `web/index.html`, and `web/app.js`. Fonts and icons require no network resources; the interface ships no raster artwork.

**Key Characteristics:**

- Compact controls with persistent labels.
- Flat surfaces separated by tone and thin rules.
- Restrained green actions and explicit text states.

## Colors

### Primary

Forest green (`accent`) identifies primary actions, keyboard focus, progress, download links, and running/completed job states. Its darker hover tone acknowledges interaction. Error red is reserved for failure text and failed job labels.

### Neutral

The pale `page` surrounds white `surface` controls and a slightly darker `canvas`. Graphite `ink` carries primary text; `muted` supports hints and metadata. `line` separates regions, while the stronger `field-border` defines editable controls. Queue selection and hover use distinct pale green fills.

**The Text State Rule.** Always pair status color with a written state; color alone does not communicate queue progress or failure.

## Typography

Use the system sans stack for interface text and the system monospace stack for logs. The frontmatter defines the reusable body, title, label, hint, and log roles. Section titles remain compact; there is no display-heading role.

The module heading uses (16px, 550), empty-state heading (17px, 550), and queue metadata (11px). Elapsed time, queue numbers, and seed metadata use tabular numerals. Text inputs, textareas, and selects become (16px) at widths up to (420px).

## Layout

The current workbench has a centered maximum width of (1480px), a fixed control column of (340px), and a flexible output region. The preview, selected-job details, and queue share that output region. Two related fields can sit side by side with a (14px) gap; field groups use a (22px) bottom margin.

At widths up to (800px), regions stack in document order: controls, preview, then queue. Horizontal padding becomes (20px). The primary form action fills its available width. The desktop preview is at least (320px) high with `height: min(44vh, 520px)`; the narrow preview is square with a (300px) minimum height. At (1600px) and above, thin outer rules bound the centered workbench.

## Elevation & Depth

There are no shadows. Pale fills, stronger field borders, and single-pixel dividers separate surfaces. Selection changes the queue row's background without lifting it.

**The Flat Surface Rule.** Use tone and borders for separation, preserving the implemented shadow-free surface hierarchy.

## Shapes

Controls and logs use the shared `radius`; the preview has slightly softer corners, and reference thumbnails have smaller corners. Queue rows remain square and edge-aligned. The local-workspace indicator is a small circular dot. Icons are inline stroked SVG.

## Components

- **Buttons:** Solid green primary, bordered white secondary, and muted text-only variants. The main form action is at least (44px) high; the compact queue action is (36px). Disabled buttons reduce opacity to (.58). Background transitions last (150ms, ease-out); reduced-motion preference removes transitions.
- **Fields:** White backgrounds, a single-pixel field border, visible labels, and green carets. The prompt resizes vertically. Disabled fields reduce opacity to (.65). Reference uploads retain the browser's file control and show ordered thumbnail rows with removal actions.
- **Focus:** A green (2px) outline with (3px) offset applies to interactive elements. Queue rows place that outline inward at (-3px) so scrolling does not clip it.
- **Preview:** A pale canvas contains either the centered SVG empty state or the selected result with `object-fit: contain`. The download link appears only after the image loads. Status, elapsed time, progress, errors, and recovery actions sit directly below it.
- **Queue:** FIFO rows show an ordinal, ellipsized prompt, seed/step metadata, and a written status. Selection uses `aria-pressed` and a pale fill. The list scrolls after (280px) height; status remains visible as the prompt contracts.
- **Disclosures and feedback:** Advanced settings and logs expand inline. Logs wrap long content and scroll after (240px) height; failure opens the log. Errors wrap and retain the form contents. A preview-load failure exposes a retry action.

## Do's and Don'ts

### Do:

- **Do** use the existing system font stacks and inline SVG icons.
- **Do** preserve visible labels, keyboard focus, and written status messages.
- **Do** let long prompts truncate in queue rows while preserving status and metadata.

### Don't:

- **Don't** add network fonts, remote decorative assets, or external UI dependencies.
- **Don't** use shadows to separate the existing flat surfaces.
- **Don't** crop the generated result to fill its preview frame.
