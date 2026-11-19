# Data sources for CivicFusion AI

## What the project needs

There are three different data jobs. Keep their data separate and connect their outputs with a complaint ID:

1. **NLP training:** complaint text and a category label.
2. **Computer-vision training:** images and visual class labels (plus bounding boxes for object detection).
3. **Live dashboard records:** new text, photo, coordinates, model results, duplicate count, status, and timestamps in MongoDB.

Do not join an NLP training CSV to a vision image dataset row by row. They usually have different complaints, locations, and IDs. The live complaint is the point where NLP and CV predictions are fused.

## Kaggle results

### Best real municipal history source: NYC 311 service requests

Kaggle mirror: [311 Service Requests NYC](https://www.kaggle.com/datasets/pablomonleon/311-service-requests-nyc)

Official source: [NYC Open Data 311 Service Requests](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9)

The official source is real city service-request data. It has issue type, a more-specific descriptor, status, agency, and geographic fields including latitude and longitude. The current official table is very large and updated; use a Kaggle sample or a filtered download for this project. NYC Open Data reports tens of millions of rows and describes the type/detail/status/location fields in its data dictionary. The current field names include `Problem`/`Problem Detail` in the site display and `complaint_type`/`descriptor` in the API.

**Fit:** useful real records for dashboard history, status charts, and mapping. For a basic NLP experiment, `descriptor` can be the text-like input and `complaint_type` the label, but the descriptor is a controlled, short field tied to the issue type. It is not the citizen's full free-text story. Do not feed the label itself into the text input, or the classifier will leak the answer.

### Closest Indian text/category source: Government of India Grievance Report

Kaggle: [Government of India: Grievance Report](https://www.kaggle.com/datasets/ayushyajnik/government-of-india-grievance-report)

The Kaggle corpus is reported as CPGRAMS public grievances and includes `subject_content_text`, `remarks_text`, and `CategoryV7`. A published study describes using those text and category fields. Verify the Kaggle data card, provenance, license, and actual rows before treating it as official operational data.

**Fit:** better text/category structure for NLP than the NYC descriptor field, but CPGRAMS covers central/state government grievances broadly; it is not a municipal-only dataset and its category hierarchy may need filtering and mapping. It does not provide the live user's photo or current GPS fix.

### BMC Kaggle competition: promising schema, provenance needs checking

Kaggle: [1.2M Complaints BMC Predict Civic Satisfaction](https://www.kaggle.com/competitions/mumbai-nagar-seva-bmc-civic-complaint-resolution-2018-2024/data)

Its page describes `text_snippet`, `complaint_category`, severity, status, and ward-related fields. However, it is a community-created competition and the page does not show an independently verifiable official BMC source for the records. Because real data matters to this project, don't use it as verified real data until its source and sample rows are checked. Its page describes a large satisfaction-prediction task, which is also different from training a municipal issue classifier.

## Computer-vision images

Kaggle: [Urban Visual Pollution Dataset](https://www.kaggle.com/datasets/abhranta/urban-visual-pollution-dataset)

A published study describes this dataset as about 9,966 street-level images across 11 visual issue classes, including garbage, road construction, potholes, broken/faded signage, and bad streetlights. It is the closest single Kaggle image source found for several of the visual classes in CivicFusion AI. Inspect the downloaded annotation files and license before training: use YOLO detection only if object bounding boxes are actually supplied; otherwise it is an image-classification dataset.

## Recommendation

- For **real map/status history**, start with a small filtered sample from NYC 311, or an official open-data source from the city you choose.
- For **NLP**, use a real complaint text/category corpus after checking its provenance and label quality. NYC's `descriptor` is a structured short description; the Government of India corpus has more narrative text but broader government categories.
- For **CV**, inspect Urban Visual Pollution annotations for the target classes and license. The currently downloaded three-class checkpoint is only a temporary demo and should be replaced after training on the chosen visual dataset.
- For **live reports**, MongoDB stores new submissions and predictions; it is separate from both model-training datasets.

## Proposed live MongoDB complaint fields

```text
complaint_id, description, image_ref, latitude, longitude,
nlp_category, vision_detections, severity, priority, department,
status, canonical_complaint_id, duplicate_count, created_at, updated_at
```

Use a single canonical complaint record for a duplicate group: increment `duplicate_count`, retain the shared issue's status and location, and link any new evidence to that record. Candidate duplicates should agree on issue category, have close text meaning, and fall within a short geographic radius and time window. Keep a staff review action for uncertain matches; never merge reports just because their wording is similar when their locations differ.

Camera and location are browser permissions. The dashboard should request them only when the user chooses to submit a report, and offer image upload plus manual map pin/address if permission is denied.
