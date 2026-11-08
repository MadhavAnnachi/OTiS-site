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
- That tag tells `app.js` never to call the Python API.
- The classical planners (exact KKT optimum, convex solver, E-guidance) therefore run in the browser through `plant.js`.
- The trained neural planner and the Training tab need the GPU server (`python main.py` in the OTiS repo). They show a short note explaining this.

## Vercel settings

| Setting | Value |
| --- | --- |
| Framework preset | Other |
| Build command | none |
| Output directory | `.` (repository root) |
| Install command | none |

Preview locally with `python -m http.server 8000` in this folder.
