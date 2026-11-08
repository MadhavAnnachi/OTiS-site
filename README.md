# OTiS web demo (Vercel)

This repository is only the hosted copy of the OTiS interface. **Don't edit the files here.**
The real project, which includes the planners, training and benchmarks, lives in the `OTiS` repository.
Every file except this README, `vercel.json` and `.gitignore` is generated from `OTiS/otis/web/`.

## Update the site

```bash
cd ../OTiS/otis
python build_site.py          # refreshes ../docs and ../../OTiS-site
cd ../../OTiS-site
git add -A && git commit -m "Sync site" && git push   # Vercel redeploys on push
```

## What runs here

This is a plain static site with no build step and no server code.

- `index.html` contains `<meta name="otis-deploy" content="static">`.
- With that tag, `app.js` sends every `/api/*` call to `demo.js`, which answers in the browser.

| File | What it holds |
| --- | --- |
| `demo/<vehicle>.bin` | The trained network for each preset vehicle (small, 584K parameters, float32). It runs in the browser in under 1 ms. |
| `demo/<vehicle>.json` | A training run recorded on an NVIDIA RTX 5070 Ti. The Training tab replays it as "simulated training", at its recorded speed. |

The classical planners (exact KKT optimum and E-guidance) come from `plant.js`. If a visitor edits the parameters away from a preset, only the classical planners run.

## Vercel settings

| Setting | Value |
| --- | --- |
| Framework preset | Other |
| Build command | none |
| Output directory | `.` (repository root) |
| Install command | none |

Preview locally with `python -m http.server 8000` in this folder.
