from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor


# TODO This whole implementation is very ugly with very ugly notation, should be refactored ASAP!


class DDIMScheduler:
    def __init__(
        self,
        *,
        num_train_timesteps: int = 1_000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        clip_sample: bool = False,  # TODO remove
        set_alpha_to_one: bool = False,
        steps_offset: int = 1,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.steps_offset = steps_offset
        self.clip_sample = clip_sample

        self.betas = torch.linspace(
            beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps,
        ).pow(2)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)

        # TODO can we remove this and replace or append to alphas?
        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )

        self.timesteps = torch.arange(0, num_train_timesteps).flip(0)

    def _get_variance(self, timestep: int, prev_timestep: int) -> float:
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance.item()

    @property
    def num_inference_steps(self) -> int:
        return len(self.timesteps)

    def set_timesteps(self, num_inference_steps: int) -> None:
        assert num_inference_steps <= self.num_train_timesteps

        self.timesteps = torch.arange(0, num_inference_steps).flip(0)

        self.timesteps *= self.num_train_timesteps // num_inference_steps
        self.timesteps += self.steps_offset

    def step(
        self,
        noise_pred: Tensor,
        timestep: int,
        latents: Tensor,
        eta: float = 0,
        use_clipped_model_output: bool = False,  # ! remove?
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        # 1. get previous step value (=t-1)
        prev_timestep = (
            timestep - self.num_train_timesteps // self.num_inference_steps
        )

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (
            latents - noise_pred * beta_prod_t.pow(1 / 2)
        ) / alpha_prod_t.pow(1 / 2)

        # 4. Clip "predicted x_0"
        # TODO remove
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** 0.5

        # TODO remove
        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            noise_pred = (
                latents - alpha_prod_t.pow(1 / 2) * pred_original_sample
            ) / beta_prod_t.pow(1 / 2)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2).pow(
            1 / 2
        ) * noise_pred

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev.pow(1 / 2) * pred_original_sample
            + pred_sample_direction
        )

        if eta > 0:
            device = noise_pred.device
            noise = torch.randn(
                noise_pred.shape, generator=generator, device=device
            )
            variance = (
                self._get_variance(timestep, prev_timestep) ** (0.5)
                * eta
                * noise
            )

            prev_sample += variance

        return prev_sample

    def add_noise(
        self, original_samples: Tensor, noise: Tensor, timesteps: Tensor,
    ) -> Tensor:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps].pow(1 / 2)
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        # TODO UGLY!!!
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]).pow(
            1 / 2
        )
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(
            original_samples.shape
        ):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples
            + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples
