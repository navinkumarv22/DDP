#pragma once
#include "embedding.h"
#include "mlp.h"

struct GPTConfig {
    int vocab_size = 50304;
    int n_embd = 768;
    int n_layers = 6;
};

struct GPT {
    GPTConfig config;
    Embedding wte;
    std::vector<MLP> mlps;
    Linear lm_head;

    Tensor last_logits;

    GPT(const GPTConfig& cfg, Device dev)
        : config(cfg),
          wte(cfg.vocab_size, cfg.n_embd, dev),
          lm_head(cfg.n_embd, cfg.vocab_size, dev) {
        for (int i = 0; i < cfg.n_layers; i++)
            mlps.emplace_back(cfg.n_embd, dev);
    }

    Tensor forward(const Tensor& idx) {
        Tensor x = wte.forward(idx);
        for (auto& m : mlps)
            x = x + m.forward(x);
        last_logits = lm_head.forward(x);
        return last_logits;
    }

    void backward(const Tensor& grad_logits) {
        Tensor g = lm_head.backward(grad_logits);
        for (int i = mlps.size() - 1; i >= 0; --i)
            g = g + mlps[i].backward(g);
        wte.backward(g);
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto add = [&](auto& v) {
            auto p = v.parameters();
            params.insert(params.end(), p.begin(), p.end());
        };
        add(wte);
        for (auto& m : mlps) add(m);
        add(lm_head);
        return params;
    }
};
