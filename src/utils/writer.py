def add_scalar_dict(metrics, writer, title, step):
    for key, value in metrics.items():
        writer.add_scalar(f"{title}_{key}", value, step)
        writer.flush()