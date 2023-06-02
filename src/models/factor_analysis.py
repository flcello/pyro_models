import torch
import numpy as np
import pyro
from tqdm import tqdm
from pyro import distributions as dist
from pyro.infer import Trace_ELBO, TraceGraph_ELBO, SVI
from pyro.optim import ClippedAdam
from src.utils import check_convergence


class FactorAnalysis:
    """Bayesian factor analysis model"""

    def __init__(
        self,
        data: torch.Tensor,
        n_factors: int = 2,
        return_values: bool = False,
        prior_scale_loc: float = 0.0,
        prior_scale_scale: float = 1.0,
        prior_z_loc: float = 0.0,
        prior_z_scale: float = 1.0,
        prior_w_loc: float = 0.0,
        prior_w_scale: float = 1.0,
        decaying_avg_baseline_beta: float = 0.95,
    ):
        """
        Args:
            data: tensor of shape (n_samples, n_features)
            n_factors: latent space dimensionality
            return_values: if True, return sampled values and params in model
                and guide
            prior_scale_loc: location parameter of LogNormal scale prior
            prior_scale_scale: scale parameter of LogNormal scale prior
            prior_z_loc: location parameter of Normal factor prior
            prior_z_scale: location parameter of Normal factor prior
            prior_w_loc: location parameter of Normal weights prior
            prior_w_scale: location parameter of Normal weights prior
            decaying_avg_baseline_beta: baseline beta parameter for
                data-dependent baseline to reduce variance
        """
        self.data = data
        self.n_factors = n_factors
        self.return_values = return_values
        self.prior_scale_loc = prior_scale_loc
        self.prior_scale_scale = prior_scale_scale
        self.prior_z_loc = prior_z_loc
        self.prior_z_scale = prior_z_scale
        self.prior_w_loc = prior_w_loc
        self.prior_w_scale = prior_w_scale
        self.decaying_avg_baseline_beta = decaying_avg_baseline_beta

        self.n_samples = data.shape[0]
        self.n_features = data.shape[1]

        self.factors_plate = pyro.plate(
            name="factors_plate",
            size=self.n_factors,
            dim=-3,
        )

        self.samples_plate = pyro.plate(
            name="samples_plate",
            size=self.n_samples,
            dim=-2,
        )

        self.features_plate = pyro.plate(
            name="features_plate",
            size=self.n_features,
            dim=-1,
        )

    def _model(self):
        with self.features_plate:
            scale = pyro.sample(
                name="scale",
                fn=dist.LogNormal(self.prior_scale_loc, self.prior_scale_scale),
            ).view(-1, 1, 1, self.n_features)

        with self.factors_plate, self.samples_plate:
            z = pyro.sample(
                name="z",
                fn=dist.Normal(self.prior_z_loc, self.prior_z_scale),
            ).view(-1, self.n_factors, self.n_samples, 1)

        with self.factors_plate, self.features_plate:
            w = pyro.sample(
                name="w", fn=dist.Normal(self.prior_w_loc, self.prior_w_scale)
            ).view(-1, self.n_factors, 1, self.n_features)

        loc = torch.einsum(
            "...knd,...knd->...nd",
            z,
            w,
        ).view(-1, 1, self.n_samples, self.n_features)

        with self.samples_plate, self.features_plate:
            y = pyro.sample(
                name="y",
                fn=dist.Normal(
                    loc=loc,
                    scale=scale,
                ),
                obs=self.data.view(-1, 1, self.n_samples, self.n_features),
            )

        if self.return_values:
            return {
                "scale": scale.squeeze(),
                "z": z.squeeze(),
                "w": w.squeeze(),
                "loc": loc.squeeze(),
                "y": y.squeeze(),
            }

    def _guide(self):
        scale_loc = pyro.param(
            name="scale_loc",
            init_tensor=torch.zeros([1, 1, self.n_features]),
        )

        scale_scale = pyro.param(
            name="scale_scale",
            init_tensor=torch.ones([1, 1, self.n_features]),
            constraint=dist.constraints.positive,
        )

        with self.features_plate:
            scale = pyro.sample(
                name="scale",
                fn=dist.LogNormal(scale_loc, scale_scale),
                infer=dict(
                    baseline={
                        "use_decaying_avg_baseline": True,
                        "baseline_beta": self.decaying_avg_baseline_beta,
                    }
                ),
            )

        z_loc = pyro.param(
            name="z_loc",
            init_tensor=torch.zeros([self.n_factors, self.n_samples, 1]),
        )

        z_scale = pyro.param(
            name="z_scale",
            init_tensor=torch.ones([self.n_factors, self.n_samples, 1]),
            constraint=dist.constraints.positive,
        )

        with self.factors_plate, self.samples_plate:
            z = pyro.sample(
                name="z",
                fn=dist.Normal(z_loc, z_scale),
                infer=dict(
                    baseline={
                        "use_decaying_avg_baseline": True,
                        "baseline_beta": self.decaying_avg_baseline_beta,
                    }
                ),
            )

        w_loc = pyro.param(
            name="w_loc",
            init_tensor=torch.zeros([self.n_factors, 1, self.n_features]),
        )

        w_scale = pyro.param(
            name="w_scale",
            init_tensor=torch.ones([self.n_factors, 1, self.n_features]),
            constraint=dist.constraints.positive,
        )

        with self.factors_plate, self.features_plate:
            w = pyro.sample(
                name="w",
                fn=dist.Normal(w_loc, w_scale),
            )

        if self.return_values:
            return {
                "scale_loc": scale_loc.squeeze(),
                "scale_scale": scale_scale.squeeze(),
                "scale": scale.squeeze(),
                "z_loc": z_loc.squeeze(),
                "z_scale": z_scale.squeeze(),
                "z": z.squeeze(),
                "w_loc": w_loc.squeeze(),
                "w_scale": w_scale.squeeze(),
                "w": w.squeeze(),
            }

    def train(
        self,
        lr: float = 5e-3,
        lrd: float = 1.0,
        num_particles: int = 10,
        convergence_window_size: int = 100,
        convergence_min_improvement: float = 1e-3,
    ):
        optimizer = ClippedAdam({"lr": lr, "lrd": lrd})

        elbo = TraceGraph_ELBO(
            num_particles=num_particles,
            vectorize_particles=True,
        )

        svi = SVI(
            model=self._model,
            guide=self._guide,
            optim=optimizer,
            loss=elbo,
        )

        self.losses = []

        def generator():
            while True:
                yield

        pbar = tqdm(generator())
        for i in pbar:
            self.losses.append(svi.step())

            relative_loss = round(self.losses[-1] / max(np.array(self.losses)), 3)
            pbar.set_description_str(f"Relative loss: {relative_loss}. Epoch")

            if check_convergence(
                losses=self.losses,
                window_size=convergence_window_size,
                min_improvement=convergence_min_improvement,
            ):
                break
