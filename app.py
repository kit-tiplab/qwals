"""Streamlit interface for the qwals linguistic distance calculator."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="qwals — Linguistic Distance",
    page_icon="🌐",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load calculator (cached so it's built only once per session)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent


@st.cache_resource(show_spinner="Loading WALS data…")
def load_calculator():
    from qwals import QwalsCalculator
    return QwalsCalculator(
        DATA_DIR / "wals-data.csv",
        DATA_DIR / "WALS_feature_order.csv",
    )


try:
    calc = load_calculator()
except Exception as exc:
    st.error(f"Could not load QwalsCalculator: {exc}")
    st.stop()

all_languages = sorted(calc.languages)

# ---------------------------------------------------------------------------
# Sidebar — global settings
# ---------------------------------------------------------------------------
st.sidebar.title("qwals settings")

method = st.sidebar.radio("Distance method", ["ordinal", "onehot"], index=0)

task_options = ["(none — all features)"] + list(calc.TASKS if hasattr(calc, "TASKS") else [])
try:
    from qwals import TASKS
    task_options = ["(none — all features)"] + list(TASKS)
except Exception:
    pass

preset = st.sidebar.selectbox("Task preset", task_options)

if preset != "(none — all features)":
    calc.use_features(preset)
else:
    calc.reset_features()

min_shared = st.sidebar.slider(
    "Min shared features",
    min_value=0,
    max_value=192,
    value=50,
    step=5,
    help="Minimum number of features two languages must share for a result to be included. "
         "Applies to nearest neighbours and transfer source suggestions. "
         "In the compare tab, triggers a warning when coverage falls below this threshold.",
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_compare, tab_nearest, tab_suggest, tab_explain, tab_matrix = st.tabs([
    "Compare two languages",
    "Nearest neighbours",
    "Suggest transfer source",
    "Explain distance",
    "Pairwise matrix",
])

# ── Tab 1: Compare two languages ────────────────────────────────────────────
with tab_compare:
    st.header("Compare two languages")
    col1, col2 = st.columns(2)
    with col1:
        lang1 = st.selectbox("Language 1", all_languages, index=all_languages.index("Polish") if "Polish" in all_languages else 0, key="cmp_l1")
    with col2:
        lang2 = st.selectbox("Language 2", all_languages, index=all_languages.index("English") if "English" in all_languages else 1, key="cmp_l2")

    show_details = st.checkbox("Show per-feature details", value=False)
    show_ci = st.checkbox("Bootstrap confidence interval", value=False)

    if st.button("Compute distance", key="btn_compare"):
        if lang1 == lang2:
            st.warning("Both languages are the same — distance is 0.")
        else:
            try:
                result = calc.distance(lang1, lang2, method=method, return_details=show_details, return_coverage=True)
                if show_details:
                    dist = result["distance"]
                    cov = result
                else:
                    dist = result["distance"]
                    cov = result

                col_a, col_b, col_c = st.columns(3)
                col_a.markdown(f"**Distance**\n\n## {dist:.4f}")
                col_b.markdown(f"**Shared features**\n\n## {cov.get('n_shared', '—')}")
                col_c.markdown(f"**Coverage**\n\n## {cov.get('coverage', 0):.1%}")
                n_shared = cov.get("n_shared", 0)
                if isinstance(n_shared, int) and n_shared < min_shared:
                    st.warning(
                        f"Only {n_shared} shared features — below your min_shared threshold of {min_shared}. "
                        "Distance estimate may be unreliable."
                    )

                if show_details:
                    st.subheader("Per-feature breakdown")
                    st.dataframe(result["details"], use_container_width=True)

                if show_ci:
                    ci = calc.distance_ci(lang1, lang2, n_bootstrap=1000, rng=42)
                    st.info(
                        f"Bootstrap CI (1 000 resamples): "
                        f"**{ci['ci_low']:.4f}** – **{ci['ci_high']:.4f}**  "
                        f"(std {ci['std']:.4f}, n_shared {ci['n_shared']})"
                    )
            except Exception as exc:
                st.error(str(exc))

# ── Tab 2: Nearest neighbours ────────────────────────────────────────────────
with tab_nearest:
    st.header("Nearest neighbours")
    nn_lang = st.selectbox("Target language", all_languages, index=all_languages.index("Polish") if "Polish" in all_languages else 0, key="nn_lang")
    nn_n = st.number_input("Number of neighbours", min_value=1, max_value=50, value=10, step=1)

    if st.button("Find neighbours", key="btn_nearest"):
        try:
            neighbours = calc.nearest(nn_lang, n=int(nn_n), method=method, min_shared=min_shared)
            if not neighbours:
                st.warning("No neighbours found with the current min_shared threshold. Try lowering it.")
            else:
                import pandas as pd
                df = pd.DataFrame(neighbours, columns=["Language", "Distance"])
                df.index = df.index + 1
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("Language")["Distance"])
        except Exception as exc:
            st.error(str(exc))

# ── Tab 3: Suggest transfer source ──────────────────────────────────────────
with tab_suggest:
    st.header("Suggest transfer source")
    st.caption("Ranks source language candidates for cross-lingual NLP transfer.")
    sug_lang = st.selectbox("Target language", all_languages, index=all_languages.index("Polish") if "Polish" in all_languages else 0, key="sug_lang")
    sug_task = st.selectbox("Task", task_options, key="sug_task")
    sug_n = st.number_input("Candidates to show", min_value=1, max_value=30, value=10, step=1)

    if st.button("Suggest", key="btn_suggest"):
        task_arg = None if sug_task == "(none — all features)" else sug_task
        try:
            suggestions = calc.suggest_transfer_source(sug_lang, task=task_arg, n=int(sug_n), min_shared=min_shared)
            import pandas as pd
            df = pd.DataFrame(suggestions)
            df.index = df.index + 1
            st.dataframe(df, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

# ── Tab 4: Explain distance ──────────────────────────────────────────────────
with tab_explain:
    st.header("Explain distance")
    st.caption("Shows which typological features contribute most to the distance between two languages.")
    col1, col2 = st.columns(2)
    with col1:
        exp_l1 = st.selectbox("Language 1", all_languages, index=all_languages.index("Polish") if "Polish" in all_languages else 0, key="exp_l1")
    with col2:
        exp_l2 = st.selectbox("Language 2", all_languages, index=all_languages.index("English") if "English" in all_languages else 1, key="exp_l2")
    exp_k = st.slider("Top-k features", min_value=5, max_value=50, value=10)

    if st.button("Explain", key="btn_explain"):
        try:
            df = calc.explain_distance(exp_l1, exp_l2, top_k=exp_k)
            st.dataframe(df, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

# ── Tab 5: Pairwise matrix ───────────────────────────────────────────────────
with tab_matrix:
    st.header("Pairwise distance matrix")
    st.caption("Select a set of languages to compute a full pairwise matrix.")

    default_langs = ["Polish", "English", "German", "Russian", "Japanese", "Korean"]
    default_langs = [l for l in default_langs if l in all_languages]

    selected = st.multiselect(
        "Languages (choose 2–30)",
        all_languages,
        default=default_langs,
        key="matrix_langs",
    )

    if st.button("Compute matrix", key="btn_matrix"):
        if len(selected) < 2:
            st.warning("Select at least 2 languages.")
        elif len(selected) > 30:
            st.warning("Please select 30 or fewer languages for performance reasons.")
        else:
            try:
                mat = calc.pairwise_matrix(selected, method=method)
                st.dataframe(mat.style.background_gradient(cmap="YlOrRd"), use_container_width=True)

                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    col_heat, col_dend = st.columns(2)
                    with col_heat:
                        fig, ax = plt.subplots(figsize=(7, 6))
                        calc.plot_heatmap(selected, method=method, ax=ax)
                        st.pyplot(fig)

                    with col_dend:
                        fig2, ax2 = plt.subplots(figsize=(7, 6))
                        calc.plot_dendrogram(selected, method=method, ax=ax2)
                        st.pyplot(fig2)
                except Exception:
                    pass  # viz extra not installed or plotting failed

            except Exception as exc:
                st.error(str(exc))
