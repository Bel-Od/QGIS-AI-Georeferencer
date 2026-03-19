# AI Georeferencer for QGIS

AI Georeferencer is a QGIS plugin for automatically georeferencing engineering plan TIFFs and PDFs. It combines OCR, OpenAI vision analysis, coordinate parsing, address-based seeding, and WMS/template matching to place a plan on the map, then lets you review and refine the result inside QGIS.

The plugin is currently primarily built around German planning documents and currently ships with a built-in geopack for North Rhine-Westphalia (NRW), including EPSG:25832 and NRW WMS services, however the users can simply modify the geopacks to adapt to their location.

## What It Does

- Reads TIFF and PDF plan files.
- Extracts text using the PDF text layer and/or Tesseract OCR.
- Uses OpenAI vision analysis to infer scale, location hints, plan type, and CRS clues.
- Builds an initial placement from OCR coordinates, project address, prior accepted results, and candidate ranking.
- Refines placement against reference map services such as aerial, topographic, cadastral/vector, or OSM-based layers.
- Writes a georeferenced GeoTIFF and loads it back into QGIS.
- Lets you accept, reject, or manually adjust the result and reuse reviewed cases for future ranking/training.

## Requirements

### Core

- QGIS `3.16` or newer
- A QGIS installation with GDAL available inside QGIS
- Internet access for OpenAI vision calls and remote WMS services

### External Dependencies

The plugin checks these from its `Setup` tab:

- `Pillow`
- `pytesseract`
- `openai`
- `PyMuPDF` for PDF input support
- `OpenCV` optional, but useful for faster image matching
- Tesseract OCR binary installed on the machine
- `OPENAI_API_KEY` configured

Notes:

- GDAL and NumPy are expected to come from QGIS and should not be separately managed through the plugin.
- The plugin currently uses the `gpt-4o` model for vision analysis.
- The code is strongly Windows-oriented, but parts of it include Linux path fallbacks.

## Installation

### Install From The Included ZIP

1. Open QGIS.
2. Go to `Plugins > Manage and Install Plugins`.
3. Click `Install from ZIP`.
4. Select [`AI_georef_plugin.zip`].
5. Install and enable `AI Georeferencer`.


## First-Time Setup

After enabling the plugin:

1. Open the `Auto Georeferencer` button from the QGIS plugins toolbar or menu.
2. Switch to the `Setup` tab.
3. Install any missing Python packages from the plugin UI.
4. Install Tesseract OCR if it is missing.
5. Confirm the Tesseract executable path if QGIS does not detect it automatically.
6. Enter and save your `OPENAI_API_KEY`.

On Windows, a common Tesseract install path is:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

For best OCR results on German plans, install the German language data (`deu`).

## How To Use The Plugin

### Basic Workflow

1. Open `Auto Georeferencer`.
2. In the `Run` tab, choose an input file.
3. Choose an output folder.
4. Select a `Reference map`.
5. For accuracy set:
   - `Scale 1:` to force the plan scale
   - `Project address:` if the site address is known
   - `Projection:` if you need a different CRS preset
6. Click `Run`.
7. Watch the live log and progress bar.
8. Review the generated georeferenced TIFF in QGIS.
9. Use the ajustment tools to refine the output.
10. The tool stores the AI output and your refinement locally so it will prove to your specific needs the more you use and train it

### Input Types

- TIFF and TIF are supported directly.
- PDF is supported when `PyMuPDF` is installed. The plugin renders the PDF, extracts the text layer when available, and supplements it with OCR.

### Reference Map Selection

The plugin can auto-detect or manually force a reference source. Built-in behavior includes:

- aerial imagery
- topographic mapping
- cadastral/vector mapping
- OSM fallback

You can also manage custom map sources from the `Manage...` button. These are stored in `map_sources.json` inside the chosen output folder.

### Project Address

The `Project address` field is important when the plan does not contain enough reliable coordinates. The plugin persists this as `project_address.json` and uses it as a high-priority seed for later runs.

### Projection / Geopacks

The plugin ships with a built-in NRW geopack:

- `EPSG:25832 - Germany/NRW (ETRS89 / UTM Zone 32N)`
- NRW DOP orthophotos
- NRW DTK topographic WMS
- NRW ALKIS cadastral/vector WMS
- NRW parcel API support

Use the `Geopacks...` button to install additional `.geopack.json` files if you want more regional presets.

## Review And Adjustment

After a run finishes, the plugin can keep improving its own placement quality through review:

- `Accept Result` marks the output as trusted and stores review metadata.
- `Reject Result` records that the result was wrong.
- `Adjust Placement` opens an interactive adjustment tool for translation, rotation, and typed corrections.

Accepted and corrected results are written into the plugin's review/library data so later runs can benefit from prior verified placements.

## Training And Ranking

The plugin contains a candidate ranking workflow:

1. Review runs with `Accept Result` or by adjusting placement.
2. Click `Export Dataset` to build `training_dataset.jsonl`.
3. Click `Retrain Ranker` to generate a new `candidate_ranker_model.json`.

This is useful if you process many similar plans from the same region or organization.

## Output Files

By default the plugin uses a writable user folder under:

```text
%USERPROFILE%\AutoGeoref\output
```

From the dialog you can choose a different output folder.

Typical outputs include:

- `*_georef.tif` or `*_rendered_georef.tif`
- `run.log`
- `last_result.json`
- `project_address.json`
- `manual_seed.json`
- `map_sources.json`
- `training_dataset.jsonl`
- `candidate_ranker_model.json`
- a `_work` folder with intermediate artifacts

Only the final georeferenced TIFF is intended as the main deliverable. The JSON and log files support reuse, diagnostics, and training.

## Recommended Operating Procedure

- Start with a clear TIFF or vector PDF if possible.
- Set the output folder per project so logs and review artifacts stay together.
- Provide a specific project address when OCR coordinates are weak or missing.
- Use the vector/cadastral reference source for fine engineering plans when aerial imagery does not match well.
- Review each result before trusting it for production use.
- Retrain the ranker only after you have collected a reasonable number of reviewed cases.

## Known Scope And Limitations

- The shipped configuration is centered on Germany, especially NRW.
- OpenAI vision features require a valid API key and network access.
- OCR quality depends heavily on scan quality and Tesseract language data.
- WMS refinement depends on reachable external map services.
- PDF support depends on `PyMuPDF`.
- The plugin has several persistence and training features, so using a stable per-project output folder is preferable to constantly changing directories.

## Repository Contents

- [`auto_georef_plugin`]
- [`auto_georef_plugin.zip`]
- [`LICENSE`]

## Uninstall

1. In QGIS, open `Plugins > Manage and Install Plugins`.
2. Disable or remove `Auto Georeferencer`.
3. Optionally use the plugin `Setup` tab cleanup tools to remove installed Python packages, saved keys, `_work` files, and generated plugin data.
4. Delete the installed `auto_georef_plugin` folder from your QGIS plugins directory if needed.
