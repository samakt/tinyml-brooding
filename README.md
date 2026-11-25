# TinyML early warning for poultry brooding (Ghana)

Research project at RAIL, KNUST. A small sensor node in each brooding pen forecasts temperature, flags sound distress and detects flooding, and warns the farmer before a chick loss. Pilot farm: Boris B's Farms.

**Status:** simulation and first training pipeline done on synthetic data. Real data collection starts after the farm visit.

## Repository map

| Folder | What is in it |
|---|---|
| `simulation/python` | Reference pen simulator (temperature, sound, flood), 28 days at 1-minute steps |
| `simulation/java` | `PenEngine`, the same model in Java, used inside AnyLogic |
| `simulation/anylogic` | Step-by-step guides to rebuild the AnyLogic model |
| `training` | `train_starter.py`: forecast, early-warning and sound models, int8 export |
| `results` | `metrics.json`, held-out plot, int8 `.tflite` models and scalers |
| `notebooks` | Colab/Kaggle notebook that clones this repo and trains |
| `docs` | Data standards and assumptions, architecture diagram, figures |
| `data` | Not committed. See `data/README.md` |

## Quick start

    make setup && source .venv/bin/activate
    make data      # about 480,000 rows, seed 11
    make train     # results in training/out

## Caveats
All results are on synthetic data labelled by the same simulator that generated it. They test the pipeline, not a farm. Values marked ASSUMPTION in the simulator are to be calibrated at the farm. See `docs/synthetic_data_standards.md`.

## Working rules
- One branch per task (`feat/…`, `fix/…`); merge by pull request.
- Never commit data, videos or decks. Use GitHub Releases or Drive.
- Changing the simulator? Regenerate with seed 11 and record the file hash in the PR.
- Commit `results/` only from a full run, with the seed and date in the message.
