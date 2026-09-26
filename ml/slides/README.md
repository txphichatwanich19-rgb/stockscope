# Slide deck source

`../stock_direction_presentation.pptx` is generated from this folder. Only needed if you want to
edit the deck's design — to just present, open the `.pptx` directly in PowerPoint/Google Slides.

## Regenerate

```bash
npm install
node make_icons.js   # renders icon PNGs into ./icons (gitignored, regenerate locally)
node build_deck.js   # writes ../stock_direction_presentation.pptx
```

Numbers on the results slides (accuracy/AUC/feature importance) are hard-coded from a run of
`../stock_direction_model.ipynb` — re-run the notebook and copy any updated numbers into
`build_deck.js` if the dataset window has moved significantly.
