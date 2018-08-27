from sklearn.preprocessing import LabelBinarizer
import numpy as np

class CustomLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super(CustomLabelBinarizer, self).transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((1-Y, Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super(CustomLabelBinarizer, self).inverse_transform(Y[:, 1], threshold)
        else:
            return super(CustomLabelBinarizer, self).inverse_transform(Y, threshold)
