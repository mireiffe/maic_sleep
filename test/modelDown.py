import torchvision.models as models

model_dict = models.__dict__

for k in model_dict.keys():
    print(f"Trying to calling {k}...", end='')
    try:
        md = getattr(models, k)
        md(pretrained=True)
        print('done!')
    except TypeError:
        print(f": Not a model")
        continue
    except ValueError:
        print(f": No available repository")

