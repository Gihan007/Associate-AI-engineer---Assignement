# Backward-compatibility shim: the old model artifact (churn_model.joblib)
# was pickled with references to `src.data_prep`. Re-export everything from
# the canonical location so joblib.load() can resolve the module.
from src.ml.data_prep import *  # noqa: F401, F403
from src.ml.data_prep import build_preprocessor, engineer_features  # noqa: F401
