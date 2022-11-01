import os.path

import Augmentor


if __name__ == '__main__':
    p = Augmentor.Pipeline('./datasets/CHASE/train/images')
    p.ground_truth(os.path.abspath('datasets/CHASE/train/1st_manual'))
    p.rotate(probability=0.8, max_left_rotation=25, max_right_rotation=25)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.6)
    p.flip_top_bottom(probability=0.5)
    p.sample(1000)

