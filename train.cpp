#include "nn/gpt.h"
#include "loss/cross_entropy.h"
#include "optim/adamw.h"

int main() {
    Device device = Device::CUDA;

    GPTConfig cfg;
    GPT model(cfg, device);

    AdamW optim(model.parameters(), 3e-4);
    CrossEntropyLoss loss_fn;

    for (int step = 0; step < 1000; step++) {
        Tensor x, y; // from DataLoaderLite

        Tensor logits = model.forward(x);
        double loss = loss_fn.forward(logits, y);

        Tensor grad_logits = loss_fn.backward();
        model.backward(grad_logits);

        optim.step();
        optim.zero_grad();
    }
}
