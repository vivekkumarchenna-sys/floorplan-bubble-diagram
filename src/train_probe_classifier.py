"""
train_probe_classifier.py - train and evaluate the learned edge-typing probe.

Reads results/probe_edge_typing_{train,val,test}.csv (see
probe_learned_edge_typing.py), trains a gradient-boosted tree classifier on
train, and reports test accuracy against the rule's own accuracy on the
identical matched-edge set (apples-to-apples comparison).
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def _find_root(start: Path) -> Path:
    for cand in (start.parent, start, *start.parents):
        if (cand / "data" / "splits").is_dir():
            return cand
    return start.parent


ROOT = _find_root(Path(__file__).resolve().parent) / "results"

FEATURES_NUM = ["overlap_px", "door_px", "opening_width", "area_a", "area_b", "centroid_dist"]
FEATURES_CAT = ["class_a", "class_b"]


def load(split):
    return pd.read_csv(ROOT / f"probe_edge_typing_{split}.csv")


def build_xy(df, encoder, fit=False):
    if fit:
        cat = encoder.fit_transform(df[FEATURES_CAT])
    else:
        cat = encoder.transform(df[FEATURES_CAT])
    num = df[FEATURES_NUM].to_numpy()
    X = np.hstack([num, cat])
    y = df["label"].to_numpy()
    return X, y


def main():
    train_df = load("train")
    val_df = load("val")
    test_df = load("test")

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train, y_train = build_xy(train_df, encoder, fit=True)
    X_val, y_val = build_xy(val_df, encoder)
    X_test, y_test = build_xy(test_df, encoder)

    best_model, best_val_acc, best_params = None, -1, None
    for max_depth in (2, 3, 4):
        # warm_start=True lets successive fit() calls extend the same
        # ensemble rather than retraining from round 0 each time, since
        # n_estimators=200 shares its first 100 boosting rounds with n_estimators=100
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=max_depth, random_state=0, warm_start=True,
        )
        for n_estimators in (100, 200):
            clf.n_estimators = n_estimators
            clf.fit(X_train, y_train)
            val_acc = accuracy_score(y_val, clf.predict(X_val))
            print(f"n_estimators={n_estimators} max_depth={max_depth} val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                # deepcopy: clf keeps mutating in place as n_estimators grows
                best_val_acc, best_model, best_params = val_acc, copy.deepcopy(clf), (n_estimators, max_depth)

    print(f"\nBest on val: n_estimators={best_params[0]} max_depth={best_params[1]} val_acc={best_val_acc:.4f}")

    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nLearned classifier test accuracy (pooled, 3-class): {test_acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    labels = sorted(set(y_test) | set(y_pred))
    print("Confusion matrix (rows=true, cols=pred), labels =", labels)
    print(confusion_matrix(y_test, y_pred, labels=labels))

    rule_acc = accuracy_score(y_test, test_df["rule_pred_type"])
    print(f"\nRule-based (existing M2 Rule 2) accuracy on the SAME matched-edge test set: {rule_acc:.4f}")
    print(classification_report(y_test, test_df["rule_pred_type"], digits=4))

    # per-class recall comparison
    print("\n=== Per-class recall comparison (test) ===")
    for cls in ["door", "shared-wall", "window"]:
        mask = y_test == cls
        if mask.sum() == 0:
            continue
        learned_recall = accuracy_score(y_test[mask], y_pred[mask])
        rule_recall = accuracy_score(y_test[mask], test_df["rule_pred_type"].to_numpy()[mask])
        print(f"  {cls:12s} n={mask.sum():5d}  learned={learned_recall:.4f}  rule={rule_recall:.4f}")

    out = pd.DataFrame({
        "metric": ["test_accuracy_pooled", "test_accuracy_pooled"],
        "method": ["learned_gbt", "rule_based_M2"],
        "value": [test_acc, rule_acc],
    })
    out.to_csv(ROOT / "probe_edge_typing_comparison.csv", index=False)
    print(f"\nSaved -> {ROOT / 'probe_edge_typing_comparison.csv'}")


if __name__ == "__main__":
    main()
