# BraTS21 dataset inventory

This inventory is metadata and validation evidence. It does not copy the source NIfTI volumes.

- Source root: `/home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21`
- Discovered patient directories: 1208
- Valid patients: 1207
- Rejected patients: 1
- Estimated source bytes: 12615269239
- Cohort hash: `5cdad558031dcf34dd4cbc2a705fa383945d08c018a9f8d80238170d447c4fdf`

## Modalities

```json
{
  "flair": 1208,
  "seg": 1208,
  "t1": 1208,
  "t1ce": 1208,
  "t2": 1208
}
```

## Geometry distributions

- Shapes: `{"[240, 240, 155]": 1207}`
- Spacing: `{"[1.0, 1.0, 1.0]": 1207}`
- Orientation: `{"LPS": 1207}`
- Segmentation labels: `{"0": 1207, "1": 1169, "2": 1206, "4": 1177}`

Rejected patients and reasons are retained in the JSON report. Patient identifiers are not sent to W&B; runs use the pseudonyms in the prepared manifest.
