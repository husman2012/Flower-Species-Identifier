import argparse
from functions import *
import cv2

parser = argparse.ArgumentParser(description = 'test')

parser.add_argument('image')
parser.add_argument('checkpoint')
parser.add_argument('--top_k', default = 5, type = int)
parser.add_argument('--category_names', default = 'cat_to_name.json')
parser.add_argument('--gpu', action='store_true', default = False)
args = parser.parse_args()

model = load_model(args.checkpoint, args.gpu)
loader, data = create_loader('flowers/train/')
model.class_to_idx = cat_to_name(data, args.category_names)
print(model.class_to_idx)

probs, name_list = predict(args.image, model, args.top_k)

print_results(probs, name_list, args.top_k)