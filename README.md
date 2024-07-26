# Volatility Estimator

Simple app and library for trade-level price cleaning and historical volatility estimation.

See [report](report/report.pdf).

Project is setup a [`poetry`](python-poetry.org/) package; see usage instructions there (or install
via `requirements.txt`, ideally in a virtual environment).

To run the batch processes, place all but the last days CSV files in `data/raw/batch` and run
(from root of directory):

```bash
python scripts/base_process_prices.py
python scripts/base_compute_volatility.py
```

To start the app for processing new data, run (from root of directory):

```bash
python app.py
```

and to simulate processing copy/move one (or all) of the last day CSV files to `data/load` (file
will be automatically deleted after being consumed, assuming no errors occur). Kill the app with
Ctrl-C.

Default environment variables are set in `.env` (in production would uncomment in `.gitignore`),
but can be overridden in the usual fashion in the terminal before or during script/app invocation.

Logs (for batch and listener) are output to `logs/app_log.json`; logs are not output to std-out.
