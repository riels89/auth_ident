# Code Authorship Identification with Contrastive Learning 
## How to use this repo
### Creating a model
1. Create a new file `src/models` with the name of the model
2. Create a new class with the name of the model and specify:
	``` python
	self.name = *name* #This is how train_model.py is going to find the model files
	self.dataset_type = ["split", "combined", "by_line"]
	```
3. Make a `create_model(self, params, index, logger)`method that returns a Keras model
4. Add the model as an import to `train_model.py`
#### Training a model
1. Create a folder in `src/models` with the name of the model spciefied in the model class.
2. Create an expirement folder in the format `EXPn-short_description-day-month-2digityear`
3. In each expirement folder create a `param_dict.json` and specify the wanted model parameters. `train_model.py` will do a grid search over all the parameters. Example:
	```json
		{
	  "optimizer": ["adam"],
	  "clipvalue": [0.5], // or "clipnorm"
	  "loss": ["contrastive"],
	  "lr": [0.00001],
	  "decay": [0.0000025],
	  "margin": [1],
	  "epochs": [4],
	  "batch_size": [64],
	  "binary_encoding": [false], // specifies whether to use binary encoding or one-hot
	  "max_code_length": [300],
	  "attention": [true, false],
	  "extra_runs": [1, 2] // Will cause each parameter combination to be run twice.
	}
		```
4. Add `trainer(model(), "short description", expirement_num, "day-month-2digityear").train()` to the end of the `train_model.py` file to train the new model or expirement. Idealy this would be replaced with a command line interface. 
5. Run`source ~/tf22/bin/activate` to activate the python environment
6. Run `export TF_KERAS=True` if you are using the attention or multi head libraries. This tells the libraries to use the keras that comes with tensorflow. It would be better to make this variable permanent so you don't have to keep setting it, but this is what I have been able to get working. 
7. Run `nohup python src/models/train_model.py &> model_name.out &`. You don't need to use nohup but you should run the model in the background to aviod the job dying early. 

### Preprocessing
The dataset currently being use is the Google Code Jam (gcj) dataset. This contains about 1.25M files from almost 100,000 authors. It should be noted that unziping the GCJ files seems to be inconsistant across machines (at least between windows and linux).
* `src/preprocessing/load_data.py` contains a number of data utilities.
* `pair_authors.py` is the file responsable for generating matching and non-matching pairs of authors for our model. 
* WIP `contrastive_author_comb.py` will only pair authors with themselvs. This was created to match the [simplified contrastive learning approach](https://arxiv.org/abs/2002.05709) Hinton proposed. 
* WIP `prepare_softmax_authors.py` will prepare the data in the format for [arcface](https://arxiv.org/abs/1801.07698) 
##### How to recreate the dataset
1. From `load_data.py` call `make_csv()` to generate a csv with all the data locations. Check the csv to make sure nothing weird with the usernames is going on. If so run `misc-scripts/fix_usernames.py`. I had an issue with this before but I do not remeber if I fixed it, so just double check. 
2. Run `create_and_save_dataset()` which will use `pair_authors.py` to generate the file pairings. Then it will pickle and save the file locations. If  the dataset were to be regenerated I would highly recommend using the method `create_file_csv.py` which was made for the `contrastive_author_comb.py`, but could be modified slightly for the `pair_authors.py` format. This will create one large csv file with the usernames and the code rather than the file locations. You will then need to modify the tf.data dataset to use their by_line function to load the data. I believe this will signifigently improve the speed of training because it will not have to load two files for each sample. 
##### tf.data datasets
* `split_dataset.py`this prepares the data and gives it in the format of `[batch_size, 2, MAX_CODE_LENGTH]`. It gives the model the files seprately which can be used in a siamese network. This has been the most used dataset version.
* `combined_dataset.py` this prepares the data in the format of `[batch_size, MAX_CODE_LENTH * 2 + 1]`. The code pair is concatenated together with 1 seperation marker to tell the model where the first code piece ends. This format seems to work well with just a simple feed forward network, probably because direct comparison can happen easier. 
* WIP`by_line_dataset.py` this preserves the code's original spatial properties by not collasping the lines. It gives the data as `[batch_size, 2, MAX_LINES, MAX_LINE_LENGTH]`. I see a lot of promise in this format if combined with a CNN because we can more easily use spatial information which would be a huge stylistic property. Unfortuantly, this has been particularly difficult to implement especially with the ragged tensors. 

## Expirements
#### Binary Cross Entropy
Models had to predict if the two given files were from the same author or different. 
* `split` and `combined` datasets with a simple feed forward NN with 6000 characters. The `combined` dataset worked well with a shallow NN and reached an accuracy of about 85%, but the split dataset and deeper NNs struggled to learn much. 
* `split` and `combined` datasets with lstms, and CNNs with 6000 did not learn much at all. 
* `combined` dataset with a shallow NN and only 2000 characters reached an accuracy of only 65%. 
#### Naive Contrastive Loss with BCE
Added a bad hacked together version of a contrastive loss combined with BCE loss. This did not learn much, but bugs that were never worked out prevented training beyond 1 epoch.
#### [Arcface](https://arxiv.org/abs/1801.07698)
The loss proposed by the Arcface paper was briefly considered, but good results from a simpler (and working) loss proposed by LeCun has put a pause on this.
#### Euclidean Contrstive Loss  ([LeCun](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)) 
This is the current iteration of our contrastive loss and has obtained great results, although with some stability issues. 
* `contrastive_bilstm_v2`. This was the first model tried with this new loss and got up to almost 90% accuracy with only 600 characters. 
* `contrastive_stacked_blistm` was tried, but unfortuantly did not fit on the character size tried (600)
* `multi_attention_bisltm` This is a bilstm with multi-headed attention stacked on top (2 heads). This model does not fit at 600 characters, but does with 300 characters. Either the size of the model or the small character limit (or both) caused a lot of stablility issues to the point where it could not learn. Lowering the learning rate to  `0.00001`and adding decay at a rate of `0.0000025` per step has seemingly fixed these issues. This is still training, but after 4 epochs it has reached almost 85% accuracy. 

To view these expirements run `nohup tensorboard --logdir models/model_name &> tensorboard.out &` on Zemenar.


## ToDo
* Get more compute!!!
* Currently accuracy is clacluated in a rather naive way, we want to compare every testing file with everyother testing file to get a more comprehensive view of our open set accuracy. This is not trivial and would likely consist of the following steps:
	1. Generate vectors for subsets of the training set code
	2. Compare every file with everyother file using a spcified threashold
	3. Search for the best threashold by repeating step 2 with different threasholds.
	4. Generate vectors for all of the validation set code
	5. Compare every file with everyother file using the optimal threashold
* Train a RF or other more traditional ML models on a closed set author problem using the generated vectors
* Making specifying and training a model a command line interface rather than having to modify `train_model.py`
* Get the `by_line_dataset` working and train it with a CNN
* Try other contrastive loss functions like Arcface. 
* Future: Using the by_line_dataset train a region proposal network and R-CNN