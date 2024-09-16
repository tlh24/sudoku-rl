
def sample_model(
    model,
    num_samples
):
    device = next(model.parameters()).device 

    