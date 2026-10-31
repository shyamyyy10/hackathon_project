# ReLoop AI ♻️

**An AI-assisted reuse-first waste sorting and recycling guide** for the NEURA X Green AI hackathon.

## What makes it different?
ReLoop AI goes beyond a waste label. It combines an image-classification model with an explainable reuse-first decision engine, asks the user about item condition, supports configurable local rules, and tracks recorded actions without making unsupported carbon-saving claims.

## Features
- Image upload and six-category waste classification: cardboard, glass, metal, paper, plastic, trash.
- Confidence-aware prediction and user correction.
- Condition-aware, explainable recommendations.
- Action log and activity dashboard.
- Explicit limitations and local-rule reminders.

## Run the app
Use Python 3.10 or 3.11 in a virtual environment.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Until a trained model exists, the recommendation flow can be tested by choosing a category manually.

## Train the image model
1. Download a waste image dataset such as TrashNet from its official repository: https://github.com/garythung/trashnet
2. Review the dataset license and provide attribution in your submission.
3. Arrange images into this structure (folder names must match exactly):

```text
data/garbage_classification/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

4. Install dependencies, then run:

```bash
python train_model.py --data_dir data/garbage_classification --epochs 8
```

5. The trained model is saved to `models/waste_classifier.keras`. Restart Streamlit and upload an image.

**Note:** Training requires downloading the dataset and pretrained MobileNetV2 weights, so an internet connection is required at setup time. TensorFlow can be large; if your laptop cannot train it, train on a compatible cloud notebook and download the resulting model file.

## Dataset and evaluation
- Start with TrashNet; its images often have simple backgrounds, so also test on your own photos.
- Keep a held-out test set separate from training and model selection.
- Report accuracy, macro F1-score, per-class recall, and a confusion matrix.
- Never claim a test score until you have actually evaluated the model.
- Use a low-confidence fallback; choose a threshold based on validation results.

## Limitations
- The six-class classifier does not identify every material type, resin code, or contamination state.
- A model prediction is not proof that an item is recyclable locally.
- Local waste rules differ. Confirm local collection guidance.
- The dashboard reports recorded user actions, not verified tonnes of waste or CO2 saved.
- The current action history is stored in `data/actions.csv` on the local app filesystem. Hosted deployments may have ephemeral storage unless a persistent database is configured.

## Suggested 90-second demo
1. Upload a clean cardboard box photo.
2. Show the prediction and confidence.
3. Confirm it is clean and intact; show the reuse-first recommendation.
4. Try a plastic item and demonstrate the local-rule choice.
5. Record actions and show the activity dashboard.
6. Use an ambiguous image to show uncertainty handling.

## Submission checklist
- [ ] Add your name/team details and a clear problem statement.
- [ ] Train and test the model, or clearly label the manual-category mode as a prototype fallback.
- [ ] Add screenshots and a short demo video.
- [ ] Deploy the Streamlit app and verify the public URL.
- [ ] Add dataset/model acknowledgements and limitations.
- [ ] Include installation steps and a reproducible GitHub repository.

## License
Choose and add a project license before publishing. Follow all dataset and pretrained-weight terms separately.
