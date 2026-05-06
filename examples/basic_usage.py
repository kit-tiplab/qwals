from qwals import QwalsCalculator

calc = QwalsCalculator(
    "wals-data.csv",
    "WALS_feature_order.csv",
)

print(calc.distance("English", "Japanese", method="ordinal"))
print(calc.distance("English", "Japanese", method="onehot"))

result = calc.distance("English", "Japanese", method="ordinal", return_details=True)
print(result["details"].head())
