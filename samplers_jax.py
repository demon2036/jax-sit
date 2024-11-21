import einops
import jax
import jax.numpy as jnp
import numpy as np
import torch


def expand_t_like_x(t, x_cur):
    """Reshape time t to broadcastable dimensions of x."""
    dims = [1] * (len(x_cur.shape) - 1)
    t = jnp.reshape(t, (t.shape[0], *dims))
    return t


def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Transform velocity prediction model to score."""
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, jnp.ones_like(xt) * -1
        sigma_t, d_sigma_t = t, jnp.ones_like(xt)
    elif path_type == "cosine":
        alpha_t = jnp.cos(t * np.pi / 2)
        sigma_t = jnp.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * jnp.sin(t * np.pi / 2)
        d_sigma_t = np.pi / 2 * jnp.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t ** 2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    """Compute diffusion coefficient."""
    return 2 * t_cur


def euler_maruyama_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatibility
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
):
    # Setup conditioning
    if cfg_scale > 1.0:
        y_null = jnp.full_like(y, 1000)

    _dtype = latents.dtype

    t_steps = jnp.linspace(1., 0.04, num_steps, dtype=jnp.float64)
    t_steps = jnp.concatenate([t_steps, jnp.array([0.], dtype=jnp.float64)])
    x_next = latents.astype(jnp.float64)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
        dt = t_next - t_cur
        x_cur = x_next
        if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
            model_input = jnp.concatenate([x_cur] * 2, axis=0)
            y_cur = jnp.concatenate([y, y_null], axis=0)
        else:
            model_input = x_cur
            y_cur = y
        kwargs = dict(y=y_cur)
        time_input = jnp.ones(model_input.shape[0], dtype=jnp.float64) * t_cur
        diffusion = compute_diffusion(t_cur)
        eps_i = jax.random.normal(jax.random.PRNGKey(i), shape=x_cur.shape)
        deps = eps_i * jnp.sqrt(jnp.abs(dt))

        # Compute drift
        v_cur = model(
            model_input.astype(_dtype), time_input.astype(_dtype), **kwargs
        )[0].astype(jnp.float64)
        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur
        if cfg_scale > 1. and guidance_low <= t_cur <= guidance_high:
            d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

        x_next = x_cur + d_cur * dt + jnp.sqrt(diffusion) * deps

    # Last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
        model_input = jnp.concatenate([x_cur] * 2, axis=0)
        y_cur = jnp.concatenate([y, y_null], axis=0)
    else:
        model_input = x_cur
        y_cur = y
    kwargs = dict(y=y_cur)
    time_input = jnp.ones(model_input.shape[0], dtype=jnp.float64) * t_cur

    # Compute drift
    v_cur = model(
        model_input.astype(_dtype), time_input.astype(_dtype), **kwargs
    )[0].astype(jnp.float64)
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and guidance_low <= t_cur <= guidance_high:
        d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur

    return mean_x


#
#
# def euler_maruyama_sampler2(
#         model,model_params,
#         latents,
#         y,
#         num_steps=20,
#         heun=False,  # not used, just for compatibility
#         cfg_scale=1.8,
#         guidance_low=0.0,
#         guidance_high=0.7,
#         path_type="linear",
# ):
#     def with_guidance(inputs):
#         x_cur, y, y_null, t_cur = inputs
#
#         model_input = jnp.concatenate([x_cur, x_cur], axis=0)
#         time_input = jnp.ones(model_input.shape[0], dtype=jnp.float64) * t_cur
#         y_cur = jnp.concatenate([y, y_null], axis=0)
#
#         v_cur = model.apply({'params':model_params},einops.rearrange(model_input, 'b c h w -> b h w c').astype(_dtype)
#                            , time_input.astype(_dtype), y_cur).astype(jnp.float64)
#
#         s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
#         d_cur = v_cur - 0.5 * diffusion * s_cur
#
#         d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
#         d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
#
#         return d_cur
#
#
#     def without_guidance(inputs):
#         x_cur, y, y_null, time_input = inputs
#         model_input = x_cur
#         time_input = jnp.ones(model_input.shape[0], dtype=jnp.float64) * t_cur
#         y_cur = y
#
#         v_cur = model.apply({'params':model_params},einops.rearrange(model_input, 'b c h w -> b h w c').astype(_dtype)
#                            , time_input.astype(_dtype), y_cur).astype(jnp.float64)
#
#         s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
#         d_cur = v_cur - 0.5 * diffusion * s_cur
#
#         return d_cur
#
#
#
#
#
#
#     # Setup conditioning
#     y_null = jnp.full_like(y, 1000)
#
#     _dtype = latents.dtype
#
#     t_steps = jnp.linspace(1., 0.04, num_steps, dtype=jnp.float64)
#     t_steps = jnp.concatenate([t_steps, jnp.array([0.], dtype=jnp.float64)])
#     x_next = latents.astype(jnp.float64)
#
#     for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
#         dt = t_next - t_cur
#         x_cur = x_next
#         diffusion = compute_diffusion(t_cur)
#         eps_i = jax.random.normal(jax.random.PRNGKey(i), shape=x_cur.shape)
#         deps = eps_i * jnp.sqrt(jnp.abs(dt))
#
#         # if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
#         #     model_input = jnp.concatenate([x_cur] * 2, axis=0)
#         #     y_cur = jnp.concatenate([y, y_null], axis=0)
#         # else:
#         #     model_input = x_cur
#         #     y_cur = y
#         # kwargs = dict(y=y_cur)
#         # time_input = jnp.ones(model_input.shape[0], dtype=jnp.float64) * t_cur
#         # diffusion = compute_diffusion(t_cur)
#         # eps_i = jax.random.normal(jax.random.PRNGKey(i), shape=x_cur.shape)
#         # deps = eps_i * jnp.sqrt(jnp.abs(dt))
#         #
#         # # Compute drift
#         # v_cur = model(
#         #     model_input.astype(_dtype), time_input.astype(_dtype), **kwargs
#         # )[0].astype(jnp.float64)
#         # s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
#         # d_cur = v_cur - 0.5 * diffusion * s_cur
#         # if cfg_scale > 1. and guidance_low <= t_cur <= guidance_high:
#         #     d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
#         #     d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
#
#         condition = (cfg_scale > 1.0) & (guidance_low <= t_cur) & (t_cur <= guidance_high)
#
#         # kwargs = dict(y=y_cur)
#
#
#         d_cur = jax.lax.cond(
#             condition,
#             with_guidance,
#             without_guidance,
#             (x_cur, y, y_null, t_cur)
#         )
#
#
#
#         x_next = x_cur + d_cur * dt + jnp.sqrt(diffusion) * deps
#
#     # Last step
#     t_cur, t_next = t_steps[-2], t_steps[-1]
#     dt = t_next - t_cur
#     x_cur = x_next
#     # if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
#     #     model_input = jnp.concatenate([x_cur] * 2, axis=0)
#     #     y_cur = jnp.concatenate([y, y_null], axis=0)
#     # else:
#     #     model_input = x_cur
#     #     y_cur = y
#
#
#     model_input = jnp.concatenate([x_cur] * 2, axis=0)
#     y_cur = jnp.concatenate([y, y_null], axis=0)
#
#
#     kwargs = dict(y=y_cur)
#     time_input = jnp.ones(model_input.shape[0], dtype=jnp.float64) * t_cur
#
#     # Compute drift
#     # v_cur = model(
#     #     model_input.astype(_dtype), time_input.astype(_dtype), **kwargs
#     # )[0].astype(jnp.float64)
#
#     v_cur = model.apply({'params': model_params}, einops.rearrange(model_input, 'b c h w -> b h w c').astype(_dtype)
#                         , time_input.astype(_dtype), y_cur).astype(jnp.float64)
#
#     s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
#     diffusion = compute_diffusion(t_cur)
#     d_cur = v_cur - 0.5 * diffusion * s_cur
#     # if cfg_scale > 1. and guidance_low <= t_cur <= guidance_high:
#     #     d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
#     #     d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
#
#     d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
#     d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
#
#     mean_x = x_cur + dt * d_cur
#
#     return mean_x


def euler_maruyama_sampler2(
        model, model_params,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatibility
        cfg_scale=1.8,
        guidance_low=0.0,
        guidance_high=0.7,
        path_type="linear",
):
    def with_guidance(inputs):
        x_cur, y, y_null, t_cur = inputs

        model_input = jnp.concatenate([x_cur, x_cur], axis=0)
        time_input = jnp.ones(model_input.shape[0], dtype=jnp.float32) * t_cur
        y_cur = jnp.concatenate([y, y_null], axis=0)

        v_cur = model.apply({'params': model_params}, einops.rearrange(model_input, 'b c h w -> b h w c').astype(_dtype)
                            , time_input.astype(_dtype), y_cur).astype(jnp.float32)

        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur

        d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

        return d_cur

    def without_guidance(inputs):
        x_cur, y, y_null, t_cur = inputs
        model_input = x_cur
        time_input = jnp.ones(model_input.shape[0], dtype=jnp.float32) * t_cur
        y_cur = y

        v_cur = model.apply({'params': model_params}, einops.rearrange(model_input, 'b c h w -> b h w c').astype(_dtype)
                            , time_input.astype(_dtype), y_cur).astype(jnp.float32)

        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur

        return d_cur

    # Setup conditioning
    y_null = jnp.full_like(y, 1000)

    _dtype = latents.dtype

    t_steps = jnp.linspace(1., 0.04, num_steps, dtype=jnp.float32)
    t_steps = jnp.concatenate([t_steps, jnp.array([0.], dtype=jnp.float32)])
    x_next = latents.astype(jnp.float32)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
        dt = t_next - t_cur
        x_cur = x_next
        diffusion = compute_diffusion(t_cur)
        eps_i = jax.random.normal(jax.random.PRNGKey(i), shape=x_cur.shape)
        deps = eps_i * jnp.sqrt(jnp.abs(dt))

        # if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
        #     model_input = jnp.concatenate([x_cur] * 2, axis=0)
        #     y_cur = jnp.concatenate([y, y_null], axis=0)
        # else:
        #     model_input = x_cur
        #     y_cur = y
        # kwargs = dict(y=y_cur)
        # time_input = jnp.ones(model_input.shape[0], dtype=jnp.float64) * t_cur
        # diffusion = compute_diffusion(t_cur)
        # eps_i = jax.random.normal(jax.random.PRNGKey(i), shape=x_cur.shape)
        # deps = eps_i * jnp.sqrt(jnp.abs(dt))
        #
        # # Compute drift
        # v_cur = model(
        #     model_input.astype(_dtype), time_input.astype(_dtype), **kwargs
        # )[0].astype(jnp.float64)
        # s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        # d_cur = v_cur - 0.5 * diffusion * s_cur
        # if cfg_scale > 1. and guidance_low <= t_cur <= guidance_high:
        #     d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
        #     d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

        condition = (cfg_scale > 1.0) & (guidance_low <= t_cur) & (t_cur <= guidance_high)

        if condition:
            print(i)
            d_cur = with_guidance((x_cur, y, y_null, t_cur))
        else:
            d_cur = without_guidance((x_cur, y, y_null, t_cur))

        # kwargs = dict(y=y_cur)
        # d_cur = jax.lax.cond(
        #     condition,
        #     with_guidance,
        #     without_guidance,
        #     (x_cur, y, y_null, t_cur)
        # )
        x_next = x_cur + d_cur * dt + jnp.sqrt(diffusion) * deps

    # return t_next


    # Last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    # if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
    #     model_input = jnp.concatenate([x_cur] * 2, axis=0)
    #     y_cur = jnp.concatenate([y, y_null], axis=0)
    # else:
    #     model_input = x_cur
    #     y_cur = y

    model_input = jnp.concatenate([x_cur] * 2, axis=0)
    y_cur = jnp.concatenate([y, y_null], axis=0)

    kwargs = dict(y=y_cur)
    time_input = jnp.ones(model_input.shape[0], dtype=jnp.float32) * t_cur

    # Compute drift
    # v_cur = model(
    #     model_input.astype(_dtype), time_input.astype(_dtype), **kwargs
    # )[0].astype(jnp.float64)

    v_cur = model.apply({'params': model_params}, einops.rearrange(model_input, 'b c h w -> b h w c').astype(_dtype)
                        , time_input.astype(_dtype), y_cur).astype(jnp.float32)

    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    # if cfg_scale > 1. and guidance_low <= t_cur <= guidance_high:
    #     d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
    #     d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
    d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur

    return mean_x


def euler_maruyama_sampler3(
        model, model_params,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatibility
        cfg_scale=1.8,
        guidance_low=0.0,
        guidance_high=0.7,
        path_type="linear",
):
    def with_guidance(inputs):
        x_cur, y, y_null, t_cur, diffusion = inputs

        model_input = jnp.concatenate([x_cur, x_cur], axis=0)
        time_input = jnp.ones(model_input.shape[0], dtype=jnp.float32) * t_cur
        y_cur = jnp.concatenate([y, y_null], axis=0)

        v_cur = model.apply({'params': model_params}, einops.rearrange(model_input, 'b c h w -> b h w c').astype(_dtype)
                            , time_input.astype(_dtype), y_cur).astype(jnp.float32)

        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur

        d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

        return d_cur

    def without_guidance(inputs):
        x_cur, y, y_null, t_cur, diffusion = inputs
        model_input = x_cur
        time_input = jnp.ones(model_input.shape[0], dtype=jnp.float32) * t_cur
        y_cur = y

        v_cur = model.apply({'params': model_params}, einops.rearrange(model_input, 'b c h w -> b h w c').astype(_dtype)
                            , time_input.astype(_dtype), y_cur).astype(jnp.float32)

        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur

        return d_cur

    # Setup conditioning
    y_null = jnp.full_like(y, 1000)

    _dtype = latents.dtype

    t_steps = jnp.linspace(1., 0.04, num_steps, dtype=jnp.float32)
    t_steps = jnp.concatenate([t_steps, jnp.array([0.], dtype=jnp.float32)])
    x_next = latents.astype(jnp.float32)

    t_curs = t_steps[:-2]
    t_nexts = t_steps[1:-1]

    def loop_body(i, x_next):
        t_next = t_nexts[i]
        t_cur = t_curs[i]



        dt = t_next - t_cur
        x_cur = x_next
        diffusion = compute_diffusion(t_cur)
        eps_i = jax.random.normal(jax.random.PRNGKey(i), shape=x_cur.shape)
        deps = eps_i * jnp.sqrt(jnp.abs(dt))


        condition = (cfg_scale > 1.0) & (guidance_low <= t_cur) & (t_cur <= guidance_high)
        d_cur = jax.lax.cond(
            condition,
            with_guidance,
            without_guidance,
            (x_cur, y, y_null, t_cur, diffusion)
        )


        x_next = x_cur + d_cur * dt + jnp.sqrt(diffusion) * deps

        return x_next

    # t_next = jax.lax.fori_loop(0, num_steps, loop_body, t_nexts[0])
    # return t_next


    x_next = jax.lax.fori_loop(0, num_steps-1, loop_body, x_next)

    # for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
    #     dt = t_next - t_cur
    #     x_cur = x_next
    #     diffusion = compute_diffusion(t_cur)
    #     eps_i = jax.random.normal(jax.random.PRNGKey(i), shape=x_cur.shape)
    #     deps = eps_i * jnp.sqrt(jnp.abs(dt))
    #
    #     # if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
    #     #     model_input = jnp.concatenate([x_cur] * 2, axis=0)
    #     #     y_cur = jnp.concatenate([y, y_null], axis=0)
    #     # else:
    #     #     model_input = x_cur
    #     #     y_cur = y
    #     # kwargs = dict(y=y_cur)
    #     # time_input = jnp.ones(model_input.shape[0], dtype=jnp.float64) * t_cur
    #     # diffusion = compute_diffusion(t_cur)
    #     # eps_i = jax.random.normal(jax.random.PRNGKey(i), shape=x_cur.shape)
    #     # deps = eps_i * jnp.sqrt(jnp.abs(dt))
    #     #
    #     # # Compute drift
    #     # v_cur = model(
    #     #     model_input.astype(_dtype), time_input.astype(_dtype), **kwargs
    #     # )[0].astype(jnp.float64)
    #     # s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    #     # d_cur = v_cur - 0.5 * diffusion * s_cur
    #     # if cfg_scale > 1. and guidance_low <= t_cur <= guidance_high:
    #     #     d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
    #     #     d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
    #
    #     condition = (cfg_scale > 1.0) & (guidance_low <= t_cur) & (t_cur <= guidance_high)
    #
    #     if condition:
    #         d_cur=with_guidance( (x_cur, y, y_null, t_cur))
    #     else:
    #         d_cur=without_guidance( (x_cur, y, y_null, t_cur))
    #
    #
    #     # kwargs = dict(y=y_cur)
    #     # d_cur = jax.lax.cond(
    #     #     condition,
    #     #     with_guidance,
    #     #     without_guidance,
    #     #     (x_cur, y, y_null, t_cur)
    #     # )
    #     x_next = x_cur + d_cur * dt + jnp.sqrt(diffusion) * deps

    # Last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    # if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
    #     model_input = jnp.concatenate([x_cur] * 2, axis=0)
    #     y_cur = jnp.concatenate([y, y_null], axis=0)
    # else:
    #     model_input = x_cur
    #     y_cur = y

    model_input = jnp.concatenate([x_cur] * 2, axis=0)
    y_cur = jnp.concatenate([y, y_null], axis=0)

    kwargs = dict(y=y_cur)
    time_input = jnp.ones(model_input.shape[0], dtype=jnp.float32) * t_cur

    # Compute drift
    # v_cur = model(
    #     model_input.astype(_dtype), time_input.astype(_dtype), **kwargs
    # )[0].astype(jnp.float64)

    v_cur = model.apply({'params': model_params}, einops.rearrange(model_input, 'b c h w -> b h w c').astype(_dtype)
                        , time_input.astype(_dtype), y_cur).astype(jnp.float32)

    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    # if cfg_scale > 1. and guidance_low <= t_cur <= guidance_high:
    #     d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
    #     d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
    d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur

    return mean_x


def euler_maruyama_sampler_fake(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatibility
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        noises=None, params=None
):
    # Setup conditioning
    if cfg_scale > 1.0:
        y_null = jnp.full_like(y, 1000)

    _dtype = latents.dtype

    t_steps = jnp.linspace(1., 0.04, num_steps, dtype=jnp.float64)
    t_steps = jnp.concatenate([t_steps, jnp.array([0.], dtype=jnp.float64)])
    x_next = latents.astype(jnp.float64)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
        dt = t_next - t_cur
        x_cur = x_next
        if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
            model_input = jnp.concatenate([x_cur] * 2, axis=0)
            y_cur = jnp.concatenate([y, y_null], axis=0)
        else:
            model_input = x_cur
            y_cur = y
        kwargs = dict(y=y_cur)
        time_input = jnp.ones(model_input.shape[0], dtype=jnp.float64) * t_cur
        diffusion = compute_diffusion(t_cur)
        # eps_i = jax.random.normal(jax.random.PRNGKey(i), shape=x_cur.shape)

        eps_i = noises[i]

        deps = eps_i * jnp.sqrt(jnp.abs(dt))

        # Compute drift
        # v_cur = model(
        #     model_input.astype(_dtype), time_input.astype(_dtype), **kwargs
        # )[0].astype(jnp.float64)

        v_cur = model.apply({'params': params}, einops.rearrange(model_input, 'b c h w -> b h w c').astype(_dtype)
                            , time_input.astype(_dtype), y_cur
                            ).astype(jnp.float64)

        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur
        if cfg_scale > 1. and guidance_low <= t_cur <= guidance_high:
            d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

        x_next = x_cur + d_cur * dt + jnp.sqrt(diffusion) * deps

    # Last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
        model_input = jnp.concatenate([x_cur] * 2, axis=0)
        y_cur = jnp.concatenate([y, y_null], axis=0)
    else:
        model_input = x_cur
        y_cur = y
    kwargs = dict(y=y_cur)
    time_input = jnp.ones(model_input.shape[0], dtype=jnp.float64) * t_cur

    # Compute drift
    # v_cur = model(
    #     model_input.astype(_dtype), time_input.astype(_dtype), **kwargs
    # )[0].astype(jnp.float64)

    v_cur = model.apply({'params': params}, einops.rearrange(model_input, 'b c h w -> b h w c').astype(_dtype),
                        time_input.astype(_dtype), y_cur,  # rngs={'dropout': rngs}
                        )

    # v_cur = model.apply({'params': params},
    #                        model_input.astype(_dtype), time_input.astype(_dtype), y_cur
    #                        ).astype(jnp.float64)

    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and guidance_low <= t_cur <= guidance_high:
        d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur

    return mean_x


def torch_to_jax(torch_x):
    return jnp.array(torch_x.detach().cpu().numpy())


def jax_to_torch(jax_x):
    return torch.from_numpy(np.array(jax_x))


def euler_maruyama_de_hacker_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatibility
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        noises=None
):
    y = torch_to_jax(y)

    # Setup conditioning
    if cfg_scale > 1.0:
        y_null = jnp.full_like(y, 1000)

    _dtype = latents.dtype

    t_steps = jnp.linspace(1., 0.04, num_steps, dtype=jnp.float64)
    t_steps = jnp.concatenate([t_steps, jnp.array([0.], dtype=jnp.float64)])

    latents = torch_to_jax(latents)

    x_next = latents.astype(jnp.float64)
    # device = x_next.device

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
        dt = t_next - t_cur
        x_cur = x_next
        if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
            model_input = jnp.concatenate([x_cur] * 2, axis=0)
            y_cur = jnp.concatenate([y, y_null], axis=0)
        else:
            model_input = x_cur
            y_cur = y

        time_input = jnp.ones(model_input.shape[0], dtype=jnp.float64) * t_cur
        diffusion = compute_diffusion(t_cur)
        # eps_i = jax.random.normal(jax.random.PRNGKey(i), shape=x_cur.shape)

        eps_i = noises[i]
        eps_i = torch_to_jax(eps_i)

        deps = eps_i * jnp.sqrt(jnp.abs(dt))

        # Compute drift
        # v_cur = model(
        #     model_input.astype(_dtype), time_input.astype(_dtype), **kwargs
        # )[0].astype(jnp.float64)
        kwargs = dict(y=jax_to_torch(y_cur))
        v_cur = model(jax_to_torch(model_input), jax_to_torch(time_input), **kwargs
                      )[0].to(torch.float64)
        v_cur = torch_to_jax(v_cur)

        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur
        if cfg_scale > 1. and guidance_low <= t_cur <= guidance_high:
            d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

        x_next = x_cur + d_cur * dt + jnp.sqrt(diffusion) * deps

    # Last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
        model_input = jnp.concatenate([x_cur] * 2, axis=0)
        y_cur = jnp.concatenate([y, y_null], axis=0)
    else:
        model_input = x_cur
        y_cur = y

    time_input = jnp.ones(model_input.shape[0], dtype=jnp.float64) * t_cur

    # Compute drift
    # kwargs = dict(y=y_cur)
    # v_cur = model(
    #     model_input.astype(_dtype), time_input.astype(_dtype), **kwargs
    # )[0].astype(jnp.float64)

    kwargs = dict(y=jax_to_torch(y_cur))
    v_cur = model(jax_to_torch(model_input), jax_to_torch(time_input), **kwargs
                  )[0].to(torch.float64)
    v_cur = torch_to_jax(v_cur)

    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and guidance_low <= t_cur <= guidance_high:
        d_cur_cond, d_cur_uncond = jnp.split(d_cur, 2, axis=0)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur

    return mean_x


def expand_t_like_x_torch(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def get_score_from_velocity_torch(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x_torch(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t ** 2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def euler_maruyama_torch_hacker_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)

    _dtype = latents.dtype

    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    noises = []

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            noises.append(eps_i)

            # compute drift
            v_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
            )[0].to(torch.float64)
            s_cur = get_score_from_velocity_torch(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

            x_next = x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y
    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
    ) * t_cur

    # compute drift
    v_cur = model(
        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
    )[0].to(torch.float64)
    s_cur = get_score_from_velocity_torch(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur

    return mean_x, noises
