import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

# 导入预安装的模块
import online_softmax_cuda as module

def run_test(module, x):
    y_custom = module.forward(x)
    y_torch = F.softmax(x, dim=-1)
    error = (y_custom - y_torch).abs().max().item()
    status = "Passed" if error < 1e-6 else "Failed"
    print("#" * 45)
    print(f"Verification: {status} (Max Error: {error:.2e})")

    print(f"\n{'Method':<25} | {'Median Time':<15}")
    print("-" * 45)
    
    methods = [
        ("PyTorch Native", lambda: F.softmax(x, dim=-1)),
        ("Custom Online (Yours)", lambda: module.forward(x))
    ]

    for label, fn in methods:
        t = benchmark.Timer(stmt='fn()', globals={'fn': fn})
        m = t.blocked_autorange()
        print(f"{label:<25} | {m.median * 1e6:>10.2f} us")
    print("-" * 45)

if __name__ == "__main__":
    # 配置参数
    BATCH_SIZE, NUM_CLASSES = 1024, 512
    x = torch.randn(BATCH_SIZE, NUM_CLASSES, device="cuda")

    run_test(module, x)