import torch
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model


def load_keras_weights(model, path):
    """Enable for loading original TensorFlow weights (.h5) to the PyTorch variant of VisualPhishNet"""

    def loss(y_true, y_pred):
        loss_value = K.maximum(y_true, y_pred)
        loss_value = K.mean(loss_value, axis=0)
        return loss_value

    keras_model = load_model(path, custom_objects={"loss": loss})
    weights = keras_model.get_weights()

    for (i, meta) in [
        (0, (0, 1)),
        (2, (2, 3)),
        (5, (4, 5)),
        (7, (6, 7)),
        (10, (8, 9)),
        (12, (10, 11)),
        (14, (12, 13)),
        (17, (14, 15)),
        (19, (16, 17)),
        (21, (18, 19)),
        (24, (20, 21)),
        (26, (22, 23)),
        (28, (24, 25)),
        (31, (26, 27)),
    ]:
        w_idx, b_idx = meta
        new_w = torch.from_numpy(weights[w_idx]).permute(3, 2, 0, 1)
        new_w = torch.nn.Parameter(new_w)

        assert (
            new_w.shape == model.layers[i].weight.shape
        ), f"{new_w.shape} vs. {model.layers[i].weight.shape}"

        model.layers[i].weight = new_w

        new_b = torch.from_numpy(weights[b_idx])
        new_b = torch.nn.Parameter(new_b)

        assert (
            new_b.shape == model.layers[i].bias.shape
        ), f"{new_b.shape} vs. {model.layers[i].bias.shape}"
        model.layers[i].bias = new_b
