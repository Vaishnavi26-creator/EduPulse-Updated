# EduPulse — Student Academic At-Risk Early Warning

## Folder structure

```text
EduPulse/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── students.csv
└── models/
    └── README.txt
```

## Add your trained model artifacts

Copy the two files exported from your Colab notebook into `models/`:

1. `student_risk_model.joblib` — rename the downloaded `student_at_risk_model.joblib` to this name.
2. `model_metadata.json` — keep this exact filename.

The model must be the complete fitted scikit-learn pipeline from your notebook, and the metadata must include `feature_columns`.

## Add your dataset

Put your actual student dataset at `data/students.csv`. Keep its column names aligned with the features used during model training. The sample CSV included here is illustrative only.

## Run locally

Open a terminal in this folder and run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Demo credentials

- Faculty: `faculty@edupulse.demo` / `Faculty@123`
- Student: `student@edupulse.demo` / `Student@123`

These are demo-only credentials. The Student dashboard currently maps the demo account to student ID `STU1001`; update this mapping for real authenticated students before using real data.

## Important

The app does not generate fallback rule-based risk predictions when the trained model is missing. It will show that the model is unavailable until both model artifacts are in place.
