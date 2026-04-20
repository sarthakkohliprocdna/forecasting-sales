# TerraForecast

Pharma territory sales forecasting. All models run in the browser — nothing to install, no credit card.

## What gets deployed

| Service | Platform | Purpose | Cost |
|---|---|---|---|
| `rxforecast-ui` | Render Static Site | Hosts the web page | Free, always on |
| `rxforecast-api` | Render Web Service | Runs Python forecast models | Free (spins down after 15min idle) |

## Deploy to Render (both services, one click)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → **Blueprint**
3. Connect the repo — Render reads `render.yaml` and creates **both services automatically**
4. Done. You get two URLs:
   - `https://rxforecast-ui.onrender.com` — share this with everyone
   - `https://rxforecast-api.onrender.com` — the backend (no need to share)

## Share with others

Send everyone this single link:
```
https://rxforecast-ui.onrender.com
```
They open it, drop their file, click Run. Everything works in their browser.

## How it works

1. User opens the UI URL
2. Drops a `.csv` or `.xlsx` file
3. 13 forecasting models run in the browser (JavaScript)
4. Full n×m matrix shown — one wMAPE per model per territory
5. Best model auto-selected per territory
6. Forecasted sales displayed + downloadable as CSV

## Models
Naive · Mean · Moving Average · Weighted MA · Linear Trend · Drift · Exp Trend · SES · Holt · Holt-Winters · ARIMA(1,1,0) · Theta · Median
