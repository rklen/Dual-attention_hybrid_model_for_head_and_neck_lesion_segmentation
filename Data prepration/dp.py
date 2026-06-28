def img_2_mask(img):
    return img.replace('img', 'mask')

def load_npfiles(imgpath):
    np_img_path = Path(imgpath)
    np_img_path = list(np_img_path.glob('*/*.npy'))
    np_img_path = list(map(lambda x:str(x), np_img_path))
    return np_img_path


class HeadandNeck():
    def __init__(self, img, transform, train_aug_both=None, transform_only_img=None):
        self.img = img
        self.transform = transform
        self.train_aug_both = train_aug_both
        self.transform_only_img = transform_only_img
    @staticmethod
    def img2mask(img):
        return img_2_mask(img)
    
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        img = np.load(self.img[idx], allow_pickle=True)
        min_val, max_val = np.percentile(img,1), np.percentile(img, 99)
        image = np.clip(img, min_val, max_val)
        normed_image = (image - image.min()) / (image.max() - image.min())
        normed_image = normed_image.astype(np.float32)

        mask = np.load(self.img2mask(self.img[idx]), allow_pickle=True)
        normed_mask = np.clip(mask, 0, 1)

        normed_mask = normed_mask.astype(np.float32)

        if self.train_aug_both:
            normed_image, normed_mask = self.train_aug_both(normed_image, normed_mask)



        if self.transform:
            normed_image = self.transform(normed_image)
            normed_mask = self.transform(normed_mask)
        if self.transform_only_img:
            normed_image = self.transform_only_img(normed_image)
        return normed_image, normed_mask
        # normed_mask = (normed_mask - normed_mask.min()) / (normed_mask.max() - normed_mask.min())


train_aug_both = transforms.Compose([ElasticDeformation(), LesionCutAndPaste(paste_count=1), MaskBoundaryJitter(max_shift=2)])
transform_only_img = transforms.Compose([transforms.GaussianBlur(3),
                                         transforms.ColorJitter(brightness=0.1, contrast=0.1)])
train_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((224,224))])
val_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])

train_path = load_npfiles('./kfoldn/img4/train')
val_path = load_npfiles('./kfoldn/img4/val')
train_dataset = HeadandNeck(train_path, train_transforms)
val_dataset = HeadandNeck(val_path, val_transforms)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=3, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=4)
# test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, drop_last=True)


print('length of Train images: ', len(train_loader))
print('length of Validation images: ', len(val_loader))
