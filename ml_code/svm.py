from sklearn.svm import SVC
from ml_config import MachineLearningConfig
from ml_validation import AccuracyValidation

config = MachineLearningConfig()

# Assuming read_training_data() returns a tuple (X, y)
image_data, target_data = config.read_training_data(config.training_data[0])

# Create the SVM model (linear kernel, probability=True for predict_proba)
svc_model = SVC(kernel='linear', probability=True, random_state=42)

# Train the model
svc_model.fit(image_data, target_data)

# Save the trained model (fix for loading issues by re-saving with current sklearn)
config.save_model(svc_model, 'SVC_model.pkl')

###############################################
# for validation and testing purposes
###############################################

validate = AccuracyValidation()

# Split validation: train-test split accuracy
validate.split_validation(svc_model, image_data, target_data, True)

# Cross validation with 3 folds
validate.cross_validation(svc_model, 3, image_data, target_data)

###############################################
# end of validation and testing
###############################################
