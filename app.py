
# import the necessary packages
import flask
import io
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import re
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = load_model()
corpus = None
def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	global corpus
	# model = ResNet50(weights="imagenet")
	pickle_off = open("classifier.pickle","rb")
	emp = pickle.load(pickle_off,encoding='latin1')
	model = emp["classifier"]
	corpus = emp["corpus"]


def prepare_review(review_text):
	# if the image mode is not RGB, convert it
	review = re.sub('[^a-zA-Z]', ' ', review_text)
	review = review.lower()
	review = review.split()
	ps = PorterStemmer()
	review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
	review = ' '.join(review)
	return review



@app.route("/", methods=["GET","POST"])
# @app.route("/predict", methods=["GET","POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# v'iew
	try:

		data = {"success": False}

		# ensure an image was properly uploaded to our endpoint
		if flask.request.method == "GET":
			# flask.jsonify({"msg":"Hello Pawan"})
			return flask.render_template("index.html")

		if flask.request.method == "POST":

			if flask.request.form.get("review_text"):
			# if True:
				# read the image in PIL format
				review_text =  flask.request.form.get("review_text")
				# image = flask.request.form["image"].read()
				print(review_text)

				# preprocess the image and prepare it for classification
				review = prepare_review(review_text)
				print(review)
				# classify the input image and then initialize the list
				# of predictions to return to the client
				corpus.append(review)
				from sklearn.feature_extraction.text import CountVectorizer
				cv = CountVectorizer(max_features=1500)
				X = cv.fit_transform(corpus).toarray()
				# print(X)
				preds = model.predict([X[-1]])
				results = preds
				data["predictions"] = results[0]

				# loop over the results and add them to the list of
				# returned predictions

				# indicate that the request was a success
				print(results)
				data["success"] = True


		# return the data dictionary as a JSON response
		return flask.render_template("index.html",data = data)
		# return flask.jsonify(data)
	except Exception as e:
		return flask.jsonify({"error":str(e)})

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run(host='0.0.0.0',port=5001,debug=True)