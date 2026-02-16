import torch
from torch import nn
import torch.nn.functional as F
import time

class SiluAndMul(nn.Module):
    """
    A custom activation layer that applies the SiLU (Sigmoid Linear Unit) activation
    function followed by element-wise multiplication with the input tensor.
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x,y = x.chunk(2, -1)
        return F.silu(x) * y
    
    if __name__ == "__main__":
        # Example usage
        layer = SiluAndMul().cuda()
        input_tensor = torch.randn(8, 4000, 8000).cuda()  # Example input tensor with shape (batch_size, sequence_length, feature_dim)

        for _ in range(10):
            _ = layer(input_tensor)  # Forward pass to test the layer

        times = []
        for _ in range(100):
            torch.cuda.synchronize()  # Ensure all previous CUDA operations are complete
            start_time = time.time()
            output_tensor = layer(input_tensor)  # Forward pass
            torch.cuda.synchronize()  # Ensure the forward pass is complete
            end_time = time.time()
            times.append(end_time - start_time)
        avg_time = sum(times) / len(times)
        print(f"Average forward pass time: {avg_time * 1000:.4f} ms")