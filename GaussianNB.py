class GaussNB(object):
    
    def __init__(self, X_model, y_model):
        self.X_model = X_model
        self.y_model = y_model
        self.class_models()
    
    # Make classes model for each attribute (lemon, apple, orange, mandarin)
    def class_models(self):
        self.labels = list(set(self.y_model))
        self.data = {label: [] for label in self.labels}
        
        #Append every data to the specified label
        # like: {'lemon': [d1, d2, d3, d4], [d5, d6, d7, d8] ...}
        for val, label in zip(self.X_model, self.y_model):
            self.data[label].append(val)
        
        self.model = {label: self.mean_std_tuple(value) for label, value in self.data.items()}
        
    #make the tuple(mean, standard_deviation) for each attribute
    def mean_std_tuple(self, train_data):
        return [(self.mean(i), self.standard_deviation(i)) for i in zip(*train_data)]
    
    def mean(self, nums):
        return np.mean(nums)
    
    def standard_deviation(self, nums):
        return np.std(nums)
    
    def calc_gaussian_prob(self, x, mean, standard_deviation):
        if standard_deviation == 0.0:
            return 1.0 if x == mean else 0.0
        
        exp = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(standard_deviation, 2))))
        return 1 / (math.sqrt(2 * math.pi) * standard_deviation) * exp
    
    # Calculate class probabilities with the input data
    def calc_class_probabilities(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                (mean, standard_deviation) = value[i]
                probabilities[label] *= self.calc_gaussian_prob(input_data[i], mean, standard_deviation)
        return probabilities
        
    def predict(self, X_test):
        predictions = []
        result = []
        for x in X_test:
            predictions.append(self.calc_class_probabilities(x));
        
        
        for prediction in predictions:
                result.append(max(prediction, key=prediction.get))
        
        return result
