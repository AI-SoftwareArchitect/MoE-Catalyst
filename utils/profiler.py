import torch
import torch.profiler

def profile_model(model, input_tensor, activities=None, on_trace_ready=None, record_shapes=True):
    activities = activities or [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=record_shapes,
        on_trace_ready=on_trace_ready,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            model(input_tensor)

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    # Eğer daha derinlemesine incelemek istersen JSON olarak kayıt da yapabilirsin
    # prof.export_chrome_trace("trace.json")
