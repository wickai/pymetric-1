
from .mbv2_searchspace import MobileNetSearchSpace


def mobilenet_v2_nas():
    search_space = MobileNetSearchSpace(
        num_classes=1000,
        small_input=False
    )
    best_individual = {'op_codes': [
        7, 7, 10, 6, 7, 6, 6, 10, 5, 3, 3, 3, 2, 10, 7, 10, 2], 'width_codes': [1, 1, 1, 1, 1, 0, 1]}
    model = search_space.get_model(
        best_individual["op_codes"], best_individual["width_codes"])
    return model
