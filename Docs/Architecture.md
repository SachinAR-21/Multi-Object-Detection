Input Image / Video
        │
        ▼
Preprocessing
        │
        ▼
YOLOv8 Model
│
├── Backbone (Feature Extraction)
├── Neck (Feature Fusion)
└── Detection Head
        │
        ▼
Non-Maximum Suppression (NMS)
        │
        ▼
Output Results
│
├── Object Labels
├── Confidence Scores
├── Bounding Boxes
└── Detected Objects Count

YOLOv8
│
├── Backbone (CSPDarknet)
├── Neck (PAN-FPN)
├── Detection Head
├── NMS
└── Output
