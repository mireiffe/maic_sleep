import torchvision.models as models

model_dict = models.__dict__

for k in model_dict.keys():
    print(f"Trying to call {k}...", end='')
    try:
        md = getattr(models, k)
        md(pretrained=True)
        print('done!')
    except TypeError: print(": Not a model")
    except ValueError: print(": No available repository")
    except NotImplementedError: print(": No available parameters")

