from icevision.all import *
import icedata

path = icedata.fridge.load_data()
print(path)
# Get the class_map, a utility that maps from number IDs to classs names
class_map = icedata.fridge.class_map()

# Randomly split our data into train/valid
data_splitter = RandomSplitter([0.8, 0.2])

# Fridge parser: provided out-of-the-box
parser = icedata.fridge.parser(data_dir=path, class_map=class_map)
train_records, valid_records = parser.parse(data_splitter)

# shows images with corresponding labels and boxes
show_records(train_records[:6], ncols=3, class_map=class_map, show=True)