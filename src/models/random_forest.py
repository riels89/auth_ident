from tensorflow import keras
import sklearn


class random_forest():

	def __init__(self):
		self.name = "random forest"
		self.dataset_type = "split"

	def create_model(self, params, index, logger):
		input = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
										  params[index]['dataset'].len_encoding), name='input')
		return sklearn.ensemble.RandomForestClassifier(**params[index])
	def set_hyperparams(self, model, input):
		pass