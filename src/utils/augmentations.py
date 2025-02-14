from torchvision import transforms
import torch


class ColorDistortion(torch.nn.Module):
    def __init__(self, s=1.0):
        super(ColorDistortion, self).__init__()
        self.s = s

    def forward(self, x):
        color_jitter = transforms.ColorJitter(
            0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s
        )
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        return color_distort(x)


class ContrastiveTransformations:
    def __init__(self, img_size):
        # transformations applied in SimCLR article
        # torchvision.
        transform = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.Grayscale(3),
            transforms.GaussianBlur(
                kernel_size=(int(0.1 * img_size)), sigma=(0.1, 0.2)
            ),
            ColorDistortion(s=1.0),
        ]
        # chose transformation with probabily 0.5 for each augmentation

        self.scripted_transforms = transforms.Compose(
            [
                transforms.RandomApply(torch.nn.ModuleList(transform), p=0.5),
            ]
        )

    def __call__(self, img):
        # it outputs a tuple, namely 2 views (augmentations) of the same image
        #   convert_tensor = transforms.ToTensor()
        #   img = convert_tensor(x)
        return (self.scripted_transforms(img), self.scripted_transforms(img))
