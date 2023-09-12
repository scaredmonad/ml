import torch
import functools

def pipe(*functions):
    def composition(*args, **kwargs):
        return functools.reduce(lambda x, f: f(x, **kwargs), functions, *args)
    return composition

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

x_ones = torch.ones_like(x_data) # retains
print(f"ones: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides dtype
print(f"rand: \n {x_rand} \n")

tn_list = [torch.randn(3, 3), torch.ones(3, 3), torch.zeros(3, 3)]
# heuristics fns
def square(tn):
    return tn ** 2
def double(tn):
    return tn * 2
composite_t = pipe(
    lambda x: [square(tn) for tn in x],
    lambda x: [double(tn) for tn in x],
    lambda x: [torch.tensor(t, dtype=torch.int64) for t in x if t.dtype is torch.float],
    torch.cat,
    # lambda x: [tn.sum() for tn in x],
)(tn_list)
print(f"concat: {composite_t}")
