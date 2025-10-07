def flat_param_specs(module, device):
    specs = []
    for p in module.parameters():
        specs.append(dict(length_elems=p.numel(), device=device, p_ref=p))
    return specs
