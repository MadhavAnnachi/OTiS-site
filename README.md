# OTiS web demo (Vercel)

Live at https://otis-site-mu.vercel.app/

This repository is only the hosted copy of the OTiS interface. **Don't edit the files here.**
The real project, which includes the planners, training and benchmarks, lives in the [OTiS](https://github.com/MadhavAnnachi/OTiS) repository.
Every file except this README, `vercel.json` and `.gitignore` is generated from `OTiS/otis/web/` by `OTiS/otis/build_site.py`.

## Update the site

With both repositories checked out side by side (`Building/OTiS` and `Building/OTiS-site`):

```bash
cd OTiS/otis
python build_site.py          # refreshes ../docs and ../../OTiS-site
cd ../../OTiS-site
git add -A && git commit -m "Sync site" && git push   # Vercel redeploys on push
```

`build_site.py` copies `index.html`, `app.js`, `plant.js`, `demo.js`, `styles.css`, `vehicles.json` and `favicon.svg`, adds the deploy tag to `index.html`, and replaces `demo/` with `OTiS/otis/web/demo/`. It leaves `README.md`, `vercel.json`, `.git` and `.vercel` alone. It needs only the Python standard library. The recorded runs in `demo/` come from `record_web_demo.py` in OTiS, which trains on the GPU; `build_site.py` does not retrain anything.

## What runs here

This is a plain static site with no build step and no server code. The interface, including the About page ("How it works"), is the same as the local OTiS app.

- `index.html` contains `<meta name="otis-deploy" content="static">`.
- With that tag, `app.js` sends every `/api/*` call to `demo.js`, which answers in the browser.
- The classical planners (exact KKT optimum and Apollo E-guidance) come from `plant.js`. It uses the same exact linear zero-order-hold physics as the Python simulator and the same stiffness limit, and works for any parameters a visitor enters.

| File | What it holds |
| --- | --- |
| `demo/<vehicle>.bin` | The trained network for each preset vehicle (Rocket, UAV, USV; small, 584K parameters, float32). `demo.js` runs it in the browser. |
| `demo/<vehicle>.json` | A training run recorded on an NVIDIA RTX 5070 Ti. The Training tab replays it as "simulated training", at its recorded speed: loss curves, log rows, evolution and the showcase plan. |
| `demo/index.json` | The list of recorded runs and the training settings they used. |

The network and the training replay are available only while the parameters match a preset. If a visitor edits the parameters away from a preset, only the classical planners run, and the page says so; training a network on new parameters needs the local OTiS server.

## Vercel settings

| Setting | Value |
| --- | --- |
| Framework preset | Other |
| Build command | none |
| Output directory | `.` (repository root) |
| Install command | none |

`vercel.json` enables clean URLs and adds basic security headers (`X-Content-Type-Options`, `Referrer-Policy`, `X-Frame-Options`).

Preview locally with `python -m http.server 8000` in this folder, then open http://localhost:8000/.
