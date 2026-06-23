#!/usr/bin/env python

import argparse
import tempfile
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pymatgen.core.composition import Composition
from matminer.featurizers.composition import ElementProperty

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

CGCNN_ROOT = Path("/Users/anitanam/cgcnn")

MODEL_PATH = Path(
    "/Users/anitanam/optical_materials_ml/notebooks/data/processed/deployable_refractive_index_model.joblib"
)

FEATURE_COLUMNS_PATH = Path(
    "/Users/anitanam/optical_materials_ml/notebooks/data/processed/deployable_feature_columns.joblib"
)

ATOM_INIT_PATH = Path(
    "/Users/anitanam/cgcnn/data/sample-regression/atom_init.json"
)

# ----------------------------------------------------------
# LOAD DEPLOYABLE MODEL
# ----------------------------------------------------------

deploy_model = joblib.load(MODEL_PATH)

feature_columns = joblib.load(
    FEATURE_COLUMNS_PATH
)

# ----------------------------------------------------------
# MAGPIE DESCRIPTORS
# ----------------------------------------------------------

ep = ElementProperty.from_preset("magpie")
magpie_cols = ep.feature_labels()

def featurize_formula(formula):

    comp = Composition(formula)

    vals = ep.featurize(comp)

    return pd.Series(
        vals,
        index=magpie_cols
    )

# ----------------------------------------------------------
# CGCNN
# ----------------------------------------------------------

sys.path.insert(0, str(CGCNN_ROOT))

from cgcnn.model import CrystalGraphConvNet
from cgcnn.data import CIFData, collate_pool

_cgcnn_model = None
_cgcnn_args = None


def load_cgcnn():

    global _cgcnn_model
    global _cgcnn_args

    if _cgcnn_model is not None:
        return

    ckpt = torch.load(
        CGCNN_ROOT / "model_best.pth.tar",
        map_location="cpu"
    )

    _cgcnn_args = ckpt["args"]

    radius = float(
        _cgcnn_args.get("radius", 8.0)
    )

    dmin = float(
        _cgcnn_args.get("dmin", 0.0)
    )

    step = float(
        _cgcnn_args.get("step", 0.2)
    )

    nbr_fea_len = int(
        (radius - dmin) / step
    ) + 1

    model = CrystalGraphConvNet(
        orig_atom_fea_len=92,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        classification=False,
    )

    model.load_state_dict(
        ckpt["state_dict"]
    )

    model.eval()

    _cgcnn_model = model


def extract_cgcnn_embedding(cif_path):

    load_cgcnn()

    radius = float(
        _cgcnn_args.get("radius", 8.0)
    )

    dmin = float(
        _cgcnn_args.get("dmin", 0.0)
    )

    step = float(
        _cgcnn_args.get("step", 0.2)
    )

    max_num_nbr = int(
        _cgcnn_args.get("max_num_nbr", 12)
    )

    embed_layer = _cgcnn_model.conv_to_fc

    captured = []

    def hook(module, inp, out):
        captured.append(
            out.detach().cpu()
        )

    with tempfile.TemporaryDirectory() as td:

        td = Path(td)

        shutil.copy(
            cif_path,
            td / "sample.cif"
        )

        shutil.copy(
            ATOM_INIT_PATH,
            td / "atom_init.json"
        )

        (
            td / "id_prop.csv"
        ).write_text(
            "sample,0.0\n"
        )

        dataset = CIFData(
            root_dir=str(td),
            max_num_nbr=max_num_nbr,
            radius=radius,
            dmin=dmin,
            step=step,
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_pool,
            num_workers=0,
        )

        handle = (
            embed_layer.register_forward_hook(
                hook
            )
        )

        try:

            with torch.no_grad():

                for inputs, _, _ in loader:

                    captured.clear()

                    _ = _cgcnn_model(*inputs)

                    emb = (
                        captured[0]
                        .squeeze(0)
                        .numpy()
                    )

                    return pd.Series(
                        emb,
                        index=[
                            f"cgcnn_emb_{i}"
                            for i in range(len(emb))
                        ]
                    )

        finally:

            handle.remove()

    raise RuntimeError(
        "Embedding extraction failed."
    )

# ----------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------

def predict_refractive_index(
    formula,
    cif_path,
):

    desc = featurize_formula(
        formula
    )

    emb = extract_cgcnn_embedding(
        cif_path
    )

    row = (
        pd.concat([desc, emb])
        .to_frame()
        .T
    )

    row = row.reindex(
        columns=feature_columns
    )

    pred = deploy_model.predict(
        row
    )[0]

    return float(pred)

# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--formula",
        required=True
    )

    parser.add_argument(
        "--cif",
        required=True
    )

    args = parser.parse_args()

    pred = predict_refractive_index(
        args.formula,
        args.cif,
    )

    print(
        f"Predicted refractive index: {pred:.4f}"
    )

if __name__ == "__main__":
    main()

