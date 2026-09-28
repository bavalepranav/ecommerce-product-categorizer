"""
Automated Product Categorizer for E-Commerce
--------------------------------------------
TF-IDF + LinearSVC text classifier with a Streamlit dashboard.

Expects a file named `ecommerce_data.csv` next to this script containing:
  * one label column  (e.g. "category")
  * one or more text columns (e.g. "title", "description")
Column names are auto-detected and can be changed from the sidebar.
Header-less two-column files (label, text) are also supported.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Product Categorizer", page_icon="🛍️", layout="wide")

DATASET_NAME = "ecommerce_data.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 5
MIN_TOTAL_ROWS = 20

LABEL_CANDIDATES = [
    "category", "product_category", "label", "class", "target",
    "category_name", "main_category", "department", "product_type", "type",
]
TITLE_CANDIDATES = ["title", "product_title", "product_name", "name", "item_name"]
DESCRIPTION_CANDIDATES = [
    "description", "product_description", "text", "product_text",
    "details", "item_description", "content",
]


class DatasetError(Exception):
    """Raised for problems with the dataset that the user can fix."""


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def find_dataset() -> Path | None:
    """Look for the dataset next to app.py, then in the working directory."""
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        candidate = base / DATASET_NAME
        if candidate.is_file():
            return candidate
    return None


def _read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV trying common encodings; skip malformed lines."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="skip", **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise DatasetError("The dataset file is empty.") from exc
        except pd.errors.ParserError as exc:
            raise DatasetError(f"The file could not be parsed as CSV: {exc}") from exc
    raise DatasetError(f"The file encoding could not be decoded: {last_error}")


def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def detect_defaults(columns) -> tuple[str | None, list[str]]:
    """Guess the label column and the text column(s) from their names."""
    norm_map = {_norm(c): c for c in columns}
    label = next((norm_map[k] for k in LABEL_CANDIDATES if k in norm_map), None)
    text_cols: list[str] = []
    for group in (TITLE_CANDIDATES, DESCRIPTION_CANDIDATES):
        hit = next((norm_map[k] for k in group if k in norm_map), None)
        if hit and hit != label and hit not in text_cols:
            text_cols.append(hit)
    return label, text_cols


@st.cache_data(show_spinner="Reading dataset...")
def load_dataframe(path_str: str, mtime: float) -> pd.DataFrame:
    """Load the CSV. `mtime` is only part of the cache key (reloads on change)."""
    df = _read_csv(path_str)

    # Semicolon / tab separated files come back as a single column.
    if df.shape[1] == 1:
        try:
            alt = _read_csv(path_str, sep=None, engine="python")
            if alt.shape[1] > 1:
                df = alt
        except Exception:
            pass

    df.columns = [str(c).strip() for c in df.columns]

    # Header-less two-column file: assume (category, description).
    if df.shape[1] == 2:
        label, text_cols = detect_defaults(df.columns)
        if label is None and not text_cols:
            df = _read_csv(path_str, header=None, names=["category", "description"])

    if df.empty:
        raise DatasetError("The dataset file contains no rows.")
    return df


# --------------------------------------------------------------------------- #
# Machine learning
# --------------------------------------------------------------------------- #
def prepare_training_data(
    df: pd.DataFrame, label_col: str, text_cols: tuple[str, ...]
) -> tuple[pd.Series, pd.Series, int]:
    """Clean the dataset and return (texts, labels, number_of_dropped_rare_classes)."""
    if label_col in text_cols:
        raise DatasetError("The label column cannot also be used as a text column.")

    work = df[[label_col, *text_cols]].copy()
    work = work.dropna(subset=[label_col])
    work[label_col] = work[label_col].astype(str).str.strip()

    if work.empty:
        raise DatasetError("No rows with a category value were found.")

    work["__text__"] = (
        work[list(text_cols)]
        .fillna("")
        .astype(str)
        .apply(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    work = work[(work[label_col] != "") & (work["__text__"] != "")]
    work = work.drop_duplicates(subset=["__text__", label_col])

    counts = work[label_col].value_counts()
    keep = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    dropped_classes = int((counts < MIN_SAMPLES_PER_CLASS).sum())
    work = work[work[label_col].isin(keep)]

    if len(work) < MIN_TOTAL_ROWS or work[label_col].nunique() < 2:
        raise DatasetError(
            f"After cleaning, only {len(work)} usable rows and "
            f"{work[label_col].nunique()} categories remain. At least "
            f"{MIN_TOTAL_ROWS} rows and 2 categories (with {MIN_SAMPLES_PER_CLASS}+ "
            "examples each) are required. Check the column selection in the sidebar."
        )
    return work["__text__"], work[label_col], dropped_classes


def build_pipeline(n_rows: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2 if n_rows >= 500 else 1,
                    sublinear_tf=True,
                    max_features=200_000,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    dual="auto",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


@st.cache_resource(show_spinner="Training the classifier (first run only)...")
def train_model(
    path_str: str, mtime: float, label_col: str, text_cols: tuple[str, ...]
) -> dict:
    """Evaluate on a held-out split, then refit on all data for serving."""
    df = load_dataframe(path_str, mtime)
    X, y, dropped_classes = prepare_training_data(df, label_col, text_cols)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:  # too few samples for a stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    eval_model = build_pipeline(len(X_train))
    eval_model.fit(X_train, y_train)
    y_pred = eval_model.predict(X_test)

    accuracy = float((np.asarray(y_pred) == y_test.to_numpy()).mean())
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report.pop("accuracy", None)
    report_df = pd.DataFrame(report).T
    summary = report_df.loc[["macro avg", "weighted avg"]]
    per_class = report_df.drop(index=["macro avg", "weighted avg"]).sort_values(
        "support", ascending=False
    )

    final_model = build_pipeline(len(X))
    final_model.fit(X, y)

    return {
        "model": final_model,
        "accuracy": accuracy,
        "macro_f1": float(summary.loc["macro avg", "f1-score"]),
        "summary": summary,
        "per_class": per_class,
        "n_rows": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_classes": int(y.nunique()),
        "dropped_classes": dropped_classes,
        "distribution": y.value_counts(),
    }


def predict_top_k(model: Pipeline, text: str, k: int = 3) -> pd.DataFrame | None:
    """Return the top-k categories with SVM decision scores, or None if no known words."""
    if model.named_steps["tfidf"].transform([text]).nnz == 0:
        return None
    scores = np.ravel(model.decision_function([text]))
    classes = np.asarray(model.classes_)
    if scores.size == 1:  # binary problem
        scores = np.array([-scores[0], scores[0]])
    order = np.argsort(scores)[::-1][:k]
    return pd.DataFrame(
        {"Category": classes[order], "Decision score": np.round(scores[order], 3)}
    )


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def render_missing_dataset() -> None:
    st.markdown(
        f"""
<div style="padding:1.5rem 1.75rem;border-radius:14px;
            border:1px solid rgba(250,90,90,.40);background:rgba(250,90,90,.08);">
  <h3 style="margin-top:0;">📂 Dataset not found</h3>
  <p style="margin-bottom:0;">
    This app looks for a file named <code>{DATASET_NAME}</code> in the project
    root (the same folder as <code>app.py</code>), but it isn't there yet.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Option A: Codespaces Explorer")
        st.markdown(
            f"""
1. Open the **Explorer** panel (left sidebar) in your Codespace.
2. Drag your CSV into the root folder, or right-click an empty area and choose **Upload...**.
3. Rename it to **`{DATASET_NAME}`**.
4. Click **Re-check for dataset** below. No restart is needed.
"""
        )
    with col_b:
        st.subheader("Option B: Terminal / GitHub")
        st.markdown(
            f"""
- **Terminal:** `cp /path/to/your_file.csv {DATASET_NAME}`
- **Commit it:** `git add {DATASET_NAME} && git commit -m "Add dataset" && git push`
- **GitHub web UI:** *Add file → Upload files* (browser limit is 25 MB).
- Files over 100 MB need [Git LFS](https://git-lfs.com).
"""
        )

    st.subheader("Expected format")
    st.markdown(
        "A CSV with a **category** column and one or more **text** columns "
        "(e.g. `title`, `description`). Column names are auto-detected and can be "
        "changed in the sidebar. Example:"
    )
    st.code(
        "title,description,category\n"
        'Wireless Mouse,"Ergonomic 2.4GHz wireless mouse with USB receiver",Electronics\n'
        'Cotton T-Shirt,"Soft crew-neck t-shirt, machine washable",Clothing\n'
        'Stainless Bottle,"Insulated 750ml water bottle keeps drinks cold",Home & Kitchen',
        language="csv",
    )

    st.button("🔄 Re-check for dataset", type="primary")


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🛍️ Automated Product Categorizer")
    st.caption(
        "TF-IDF features + Linear SVM. Describe a product and get its predicted "
        "e-commerce category."
    )

    # 1. Locate and load dataset --------------------------------------------- #
    path = find_dataset()
    if path is None:
        render_missing_dataset()
        st.stop()

    mtime = path.stat().st_mtime
    try:
        df = load_dataframe(str(path), mtime)
    except DatasetError as exc:
        st.error(f"❌ Could not load `{DATASET_NAME}`: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Unexpected error while reading `{DATASET_NAME}`: {exc}")
        st.stop()

    # 2. Column mapping ------------------------------------------------------ #
    columns = list(df.columns)
    label_default, text_default = detect_defaults(columns)
    with st.sidebar:
        st.header("⚙️ Dataset")
        st.markdown(f"**File:** `{path.name}`  \n**Rows:** {len(df):,}")
        label_col = st.selectbox(
            "Category (label) column",
            columns,
            index=columns.index(label_default) if label_default in columns else 0,
        )
        text_options = [c for c in columns if c != label_col]
        text_cols = st.multiselect(
            "Text column(s)",
            text_options,
            default=[c for c in text_default if c in text_options],
            help="Selected columns are concatenated into one text per product.",
        )

    if not text_cols:
        st.info("👈 Select at least one text column in the sidebar to train the model.")
        st.stop()

    # 3. Train (cached) ------------------------------------------------------ #
    try:
        art = train_model(str(path), mtime, label_col, tuple(text_cols))
    except DatasetError as exc:
        st.error(f"❌ {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error("❌ Training failed. Check the column selection or dataset contents.")
        st.exception(exc)
        st.stop()

    model = art["model"]
    model_key = (str(path), mtime, label_col, tuple(text_cols))

    if art["dropped_classes"]:
        st.warning(
            f"{art['dropped_classes']} category(ies) with fewer than "
            f"{MIN_SAMPLES_PER_CLASS} examples were excluded from training."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training rows", f"{art['n_rows']:,}")
    m2.metric("Categories", f"{art['n_classes']:,}")
    m3.metric("Test accuracy", f"{art['accuracy']:.1%}")
    m4.metric("Macro F1", f"{art['macro_f1']:.3f}")

    st.divider()

    # 4. Interactive prediction --------------------------------------------- #
    st.subheader("🔎 Categorize a product")
    user_text = st.text_area(
        "Product description",
        height=130,
        placeholder="e.g. Noise-cancelling over-ear Bluetooth headphones with 30-hour battery life",
    )

    if st.button("Categorize", type="primary"):
        cleaned = user_text.strip()
        if not cleaned:
            st.session_state.pop("result", None)
            st.warning("Please enter a product description first.")
        else:
            st.session_state["result"] = {
                "key": model_key,
                "text": cleaned,
                "top": predict_top_k(model, cleaned),
            }

    result = st.session_state.get("result")
    if result and result["key"] == model_key:
        top = result["top"]
        if top is None:
            st.warning(
                "None of the words in that description are known to the model "
                "(after removing common English stop words). Try a more descriptive text."
            )
        else:
            st.success(f"**Predicted category:** {top.iloc[0]['Category']}")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Top matches** (higher score = stronger match)")
                st.dataframe(top, hide_index=True)
            with c2:
                st.bar_chart(top.set_index("Category")["Decision score"])

    # 5. Metrics ------------------------------------------------------------- #
    with st.expander("📊 Classification report"):
        st.caption(
            f"Evaluated on a held-out {art['n_test']:,}-row test set "
            f"({art['n_train']:,} rows used for training). The deployed model is "
            "then refit on all rows."
        )
        a, b = st.columns(2)
        a.metric("Accuracy", f"{art['accuracy']:.3f}")
        b.metric("Weighted F1", f"{art['summary'].loc['weighted avg', 'f1-score']:.3f}")

        st.markdown("**Averages**")
        st.dataframe(art["summary"].round(3))

        st.markdown("**Per-category metrics**")
        st.dataframe(art["per_class"].round(3))

        st.markdown("**Category distribution**")
        st.bar_chart(art["distribution"])


main()