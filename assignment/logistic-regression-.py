import math


def sigmoid(z):
    """Return the sigmoid probability for a real-valued input."""
    return 1 / (1 + math.exp(-z))


def predict_class(z):
    """Predict class label (0/1) from a logit value using sigmoid."""
    return 1 if sigmoid(z) >= 0.5 else 0


def evaluate(y_true, y_pred):
    """Compute accuracy, false positives, and false negatives."""
    total = len(y_true)
    correct = sum(1 for actual, pred in zip(y_true, y_pred) if actual == pred)
    false_positives = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 0 and pred == 1)
    false_negatives = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 1 and pred == 0)

    return {
        "accuracy": (correct / total) * 100 if total else 0.0,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def display_report(y_true, y_pred, z_values):
    """Print probabilities, predicted classes, and evaluation metrics."""
    probabilities = [sigmoid(z) for z in z_values]
    predicted_classes = [predict_class(z) for z in z_values]
    metrics = evaluate(y_true, y_pred)

    print(f"Probabilities   : {probabilities}")
    print(f"Predicted Classes: {predicted_classes}")
    print(f"Accuracy        : {metrics['accuracy']:.2f}%")
    print(f"False Positives : {metrics['false_positives']}")
    print(f"False Negatives : {metrics['false_negatives']}")


z_values = [2.5, -1.2, 0.0, 3.1, -2.8]
y_true = [1, 0, 1, 1, 0]
y_pred = [predict_class(z) for z in z_values]

display_report(y_true, y_pred, z_values)
