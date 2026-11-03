
import numpy as np 

def cosine_similarity(a, b):
    """Trả về cosine giữa hai vector a, b (float). Vector 0 -> 0.0."""
    # Viết code của bạn ở đây
    x = np.asarray(a, dtype = float)
    y = np.asarray(b, dtype = float)

    dot_pro = np.sum(x * y)

    vec1 = np.sqrt(np.sum(x*x))

    vec2 = np.sqrt(np.sum(y*y))

    if vec1 * vec2 == 0:
        return 0.0

    return dot_pro / (vec1 * vec2)
