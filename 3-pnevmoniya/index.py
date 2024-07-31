#%%
import numpy as np 
import pandas as pd 
from fastai.vision.all import *


# READ
path = Path('train')
file=get_image_files(path)


# DATALOADER 
transports = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = RandomSplitter(valid_pct=0.1, seed=4),
    get_y = parent_label,
    item_tfms = Resize(224)
)
dls = transports.dataloaders(path)


# IMAGE_LEARNING
learn = vision_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(3) 

learn1 = vision_learner(dls, resnet50, metrics=accuracy)
learn1.fine_tune(3) 


# GRAPHIC TEST
inter = ClassificationInterpretation.from_learner(learn)
inter.plot_confusion_matrix()

inter = ClassificationInterpretation.from_learner(learn1)
inter.plot_confusion_matrix()


# SAVE RESULTS
def get_test_image_path(index):
    return Path(f"test/{df_sample.loc[index, 'id']}")

def predict_image(index):
    return learn.predict(PILImage.create(get_test_image_path(index)))[2][1]

df_sample = pd.read_csv('sample_solution.csv')
df_sample['labels'] = [np.array(np.round(predict_image(i))) for i in range(len(df_sample))]
df_sample.to_csv('submission.csv', index=False)
# %%
