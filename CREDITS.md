# Credits and third-party material

## Authors

- **DeepDAS** (Deep Learning course project, Tel Aviv University, 2026): Noa Sapir, Asaf Zini, Neta Shevo.
- **SimStudio** (Final Project No. 3221, School of Electrical Engineering, Tel Aviv University, 2025–2026: *Optical Fibers for Smart Cities*): Noa Sapir and Nitai Dobrecki; supervisor: Khen Cohen.

## Reused project code
The simulator, sensor models, tracker and Z-score anomaly model of SimStudio are included **unchanged**:

- `src/deepdas/fp/`
- `src/deepdas/fp_anomaly_model/`
- `src/deepdas/batches/populate_vehicles.py`

## Data sources

- **Road network.** The road network of the North Tel Aviv simulation regions and of the Klausner Street site is derived from **OpenStreetMap**. © OpenStreetMap contributors. The data is available under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/). This attribution applies to the scenario files in the data package and to the map images in `docs/assets/simstudio/`.
- **Klausner recording.** The real DAS and camera recording on Klausner Street was made with the SimStudio project.

## Images

- **Logo.** The *Optical Fibers for Smart Cities* logo is the SimStudio project logo, used here by its authors.
- **Background images.** The images in `docs/assets/simstudio/` come from the SimStudio final presentation, by the same authors.

## Software
The software is listed in `requirements.txt` (NumPy, pandas, PyArrow, PyTorch, scikit-learn, SciPy, python-calamine, openpyxl, Matplotlib, tabulate). Each package is used under its own license, and none is redistributed in this repository.

## License of this repository
None has been chosen yet; until the authors add one, all rights are reserved.
