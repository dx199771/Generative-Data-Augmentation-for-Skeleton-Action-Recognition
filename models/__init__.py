def get_model(cfg):
    if cfg.model == "mdm_condition":
        from .diffusion import GaussianDiffusion
        from .cgn import CGN
        from .sampler import get_named_beta_schedule

        betas = get_named_beta_schedule('cosine', 1000, 1.0)
        model = CGN(
            input_feats  = cfg.input_feats,
            njoints      = cfg.njoints,
            nfeats       = cfg.nfeats,
            num_actions  = cfg.num_actions,
            training     = cfg.training,
            dataset      = cfg.dataset_name,
            dropout      = cfg.dropout_inf,
        ).cuda()
        diffusion = GaussianDiffusion(betas)
        return model, diffusion

    raise NotImplementedError(f"Unknown model: {cfg.model}")