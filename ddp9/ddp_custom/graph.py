import os, torch
from train import TinyLM , TinyLMConfig 

def main():
    model_cfg = TinyLMConfig(n_embd=256, vocab_size=50304, block_size=512)
    model = TinyLM(model_cfg)     
    model.eval()

    # torchviz autograd graph
    try:
        from torchviz import make_dot
        model = model.to('cpu')
        example = torch.randn(2, 10)   
        target  = torch.randn(2, 10)
        out  = model(example)
        loss = torch.nn.functional.mse_loss(out, target)
        dot  = make_dot(loss, params=dict(model.named_parameters()))
        os.makedirs("outputs", exist_ok=True)
        dot.render("outputs/autograd_graph", format="pdf")
        print("Wrote outputs/autograd_graph.pdf")
    except Exception as e:
        print("torchviz export failed:", e)

if __name__ == "__main__":
    main()
