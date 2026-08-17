
def predict(age: int, income: float, credit_score: int):
    """
    Simple prediction logic for our deployment lab.

    This will later be replaced by a real ML model.
    """

    risk_score = 0

    if credit_score < 600:
        risk_score += 2

    if income < 30000:
        risk_score += 1

    if age < 25:
        risk_score += 1

    prediction = 1 if risk_score >= 2 else 0

    return {
        "prediction": prediction,
        "risk_score": risk_score,
    }
