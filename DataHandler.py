from six.moves.urllib.request import urlretrieve
from matplotlib.pyplot import imread
import numpy as np
import tarfile
import sys
import os

URL = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
np.random.seed(133)


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(data_root, filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(data_root, filename)

    if force or not os.path.exists(dest_filename):

        if not os.path.exists(data_root):
            os.mkdir(data_root)

        print('Attempting to download:', filename)
        filename, _ = urlretrieve(URL + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
            'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename


def maybe_extract(data_root, filename, num_classes, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


def load_letter(folder, min_num_images, shape=(28, 28)):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = []

    print(folder)

    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = imread(image_file).astype(np.float32)
            mean = np.mean(image_data)
            stddev = np.std(image_data)
            image_data = (image_data - mean) / stddev  # Standardize image
            # image_data = (ndimage.imread(image_file).astype(float) -
            #               pixel_depth / 2) / pixel_depth
            if image_data.shape != shape:
                raise Exception(f"Unexpected image shape: {image_data.shape}")

            dataset.append(image_data)

        except IOError as e:
            print(f"Could not read:{image_file}: e- it\'s ok skipping.")

    if len(dataset) < min_num_images:
        raise Exception(f"Many fewer images than expected: {len(dataset)} < {min_num_images}")

    dataset = np.asarray(dataset)

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_tfRecord(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names
