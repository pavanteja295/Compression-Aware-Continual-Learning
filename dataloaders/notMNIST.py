class notMNIST_(torch.utils.data.Dataset):

    def __init__(self, root, task_num, num_samples_per_class, train,transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = "https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/master/data/notMNIST.zip"
        self.filename = 'notMNIST.zip'
        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()
		import pdb; pdb.set_trace()
        if self.train:
            fpath = os.path.join(root, 'notMNIST', 'Train')

        else:
            fpath = os.path.join(root, 'notMNIST', 'Test')


        X, Y = [], []
        folders = os.listdir(fpath)

        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    X.append(np.array(Image.open(img_path).convert('RGB')))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    print("File {}/{} is broken".format(folder, ims))
        self.data = np.array(X)
        self.targets = Y

        self.num_classes = len(set(self.targets))


        if num_samples_per_class:
            x, y [], [], [], []
            for l in range(self.num_classes):
                indices_with_label_l = np.where(np.array(self.targets)==l)

                x_with_label_l = [self.data[item] for item in indices_with_label_l[0]]

                # If we need a subset of the dataset with num_samples_per_class we use this and then concatenate it with a complete dataset
                shuffled_indices = np.random.permutation(len(x_with_label_l))[:num_samples_per_class]
                x_with_label_l = [x_with_label_l[item] for item in shuffled_indices]
                y_with_label_l = [l]*len(shuffled_indices)

                x.append(x_with_label_l)
                y.append(y_with_label_l)

            self.data = np.array(sum(x,[]))
            self.labels = sum(y,[])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)#.convert('RGB')
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        """Download the notMNIST data if it doesn't exist in processed_folder already."""

        import errno
        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()
