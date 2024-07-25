from ray.rllib.algorithms.dreamerv3.dreamerv3_catalog import DreamerV3Catalog
from ray.rllib.core.models.base import Model
from ray.rllib.utils import override


import gymnasium as gym

from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.base import Encoder, Model
from ray.rllib.utils import override


class DreamerV3SudokuCatalog(Catalog):
    '''
    TODO: build this if using 2D or 3D observation spaces, since only vector
        or 3d 64x64 works out of box
    '''
    def __init__(self, observation_space, action_space, model_size: str = 'XS'):
        model_config_dict = {"model_size" : model_size}
        super().__init__(observation_space,action_space, model_config_dict)

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )

        self.model_size = self._model_config_dict["model_size"]
        self.is_img_space = len(self.observation_space.shape) in [2, 3]
        self.is_gray_scale = (
            self.is_img_space and len(self.observation_space.shape) == 2
        )

        # TODO (sven): We should work with sub-component configurations here,
        #  and even try replacing all current Dreamer model components with
        #  our default primitives. But for now, we'll construct the DreamerV3Model
        #  directly in our `build_...()` methods.

    @override(Catalog)
    def build_encoder(self, framework: str) -> Encoder:
        """Builds the World-Model's encoder network depending on the obs space."""
        if framework != "tf2":
            raise NotImplementedError

        if self.is_img_space:
            from ray.rllib.algorithms.dreamerv3.tf.models.components.cnn_atari import (
                CNNAtari,
            )

            return CNNAtari(model_size=self.model_size)
        else:
            from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP

            return MLP(model_size=self.model_size, name="vector_encoder")

    def build_decoder(self, framework: str) -> Model:
        """Builds the World-Model's decoder network depending on the obs space."""
        if framework != "tf2":
            raise NotImplementedError

        if self.is_img_space:
            from ray.rllib.algorithms.dreamerv3.tf.models.components import (
                conv_transpose_atari,
            )

            return conv_transpose_atari.ConvTransposeAtari(
                model_size=self.model_size,
                gray_scaled=self.is_gray_scale,
            )
        else:
            from ray.rllib.algorithms.dreamerv3.tf.models.components import (
                vector_decoder,
            )

            return vector_decoder.VectorDecoder(
                model_size=self.model_size,
                observation_space=self.observation_space,
            )


    

