import nltk, numpy, tflearn, random, json, pickle, sys, os, glob

stemmer = nltk.stem.lancaster.LancasterStemmer()

if "retrain" in sys.argv[1:]:
    print("Deleting model and data")
    if os.path.exists("data.pickle"):
        os.remove("data.pickle")
    fileList = glob.glob("model.tfl.*")
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Couldn't remove file {0}".format(filePath))

with open("training.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            docs_x.append(w)
            docs_y.append(intent['tag'])
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in ["?", "!", "."]]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        stems = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in stems:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        self.val_acc_thresh = val_acc_thresh

    def on_epoch_end(self, training_state):
        if training_state.epoch < 1000: return
        if training_state.acc_value is None: return
        if training_state.acc_value > self.val_acc_thresh:
            raise StopIteration


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.9999)

if os.path.exists("model.tfl.meta"):
    model.load("model.tfl")
else:
    try:
        model.fit(training, output, n_epoch=4000, batch_size=8, show_metric=True, callbacks=early_stopping_cb)
    except StopIteration:
        print("Stopping early, returning...")
    model.save("model.tfl")


def create_bag(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for stem in s_words:
        for i, w in enumerate(words):
            if w == stem:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    while True:
        request = input("> ")
        if request.lower() == "quit" or request.lower() == "":
            break
        results = model.predict([create_bag(request, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] < 0.70:
            print(results)
            print("I don't know how to answer this question.")
        else:
            for t in data['intents']:
                if t["tag"] == tag:
                    responses = t['responses']

            print(random.choice(responses))


chat()
