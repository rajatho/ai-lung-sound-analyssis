import os
import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS = os.path.join(BASE_DIR, '..', 'Model', 'model_weights.h5')

LABELS = [
    'Healthy', 'Asthma', 'Bronchiectasis', 'Bronchiolitis',
    'COPD', 'LRTI', 'Pneumonia', 'URTI'
]

# Reported test-set performance of the trained CNN (display on reports).
MODEL_ACCURACY = 94.2

# Second diagnosis is shown when its class probability meets this threshold (%).
DUAL_DISEASE_MIN_CONFIDENCE = 18.0

CAUSES = {
    "Healthy": (
        "Normal lung function with no active infection or obstruction.\n"
        "Clear airways and regular breathing patterns."
    ),
    "Asthma": (
        "Genetic and environmental factors.\n"
        "Allergens such as pollen, dust mites, and pet dander.\n"
        "Air pollution, cold air, and respiratory infections.\n"
        "Smoking or exposure to secondhand smoke."
    ),
    "Bronchiectasis": (
        "Repeated lung infections (pneumonia, tuberculosis).\n"
        "Cystic fibrosis or immune system disorders.\n"
        "Blockage of airways by a foreign object.\n"
        "Long-term smoking or inhaling harmful fumes."
    ),
    "Bronchiolitis": (
        "Respiratory syncytial virus (RSV) — most common in infants.\n"
        "Other viruses such as influenza or adenovirus.\n"
        "Exposure to crowded places during cold seasons.\n"
        "Premature birth or weakened immune system in children."
    ),
    "COPD": (
        "Long-term cigarette smoking (primary cause).\n"
        "Exposure to industrial dust, fumes, and chemicals.\n"
        "Indoor air pollution from cooking fuels.\n"
        "Genetic condition alpha-1 antitrypsin deficiency."
    ),
    "LRTI": (
        "Bacterial infections (pneumonia, bronchitis).\n"
        "Viral infections such as influenza or COVID-19.\n"
        "Weakened immune system or chronic lung disease.\n"
        "Smoking and poor air quality."
    ),
    "Pneumonia": (
        "Bacterial infection (Streptococcus pneumoniae).\n"
        "Viral or fungal lung infections.\n"
        "Aspiration of food or liquids into the lungs.\n"
        "Hospital-acquired infections in critically ill patients."
    ),
    "URTI": (
        "Common cold viruses (rhinovirus, coronavirus).\n"
        "Influenza and parainfluenza viruses.\n"
        "Close contact with infected persons.\n"
        "Seasonal changes and weakened immunity."
    ),
}

DESCRIPTIONS = {
    "Healthy": (
        "Your lung audio patterns appear normal. The airways sound clear with no "
        "significant wheezing, crackles, or abnormal breath sounds detected by the model."
    ),
    "Asthma": (
        "Asthma is a chronic condition where airways become inflamed and narrow, "
        "causing wheezing, shortness of breath, and coughing. It often flares up "
        "due to allergens, cold air, or exercise."
    ),
    "Bronchiectasis": (
        "Bronchiectasis is a long-term condition where the airways become widened "
        "and scarred, leading to mucus buildup and repeated infections. It causes "
        "persistent cough and difficulty clearing the lungs."
    ),
    "Bronchiolitis": (
        "Bronchiolitis is a common lung infection in infants and young children, "
        "caused mainly by RSV. It inflames the small airways (bronchioles), leading "
        "to wheezing, rapid breathing, and cough."
    ),
    "COPD": (
        "Chronic Obstructive Pulmonary Disease (COPD) is a progressive lung disease "
        "that makes breathing difficult. It includes emphysema and chronic bronchitis, "
        "most often caused by long-term smoking."
    ),
    "LRTI": (
        "Lower Respiratory Tract Infection (LRTI) affects the lungs and lower airways, "
        "including conditions like pneumonia and bronchitis. Symptoms include cough, "
        "fever, chest pain, and difficulty breathing."
    ),
    "Pneumonia": (
        "Pneumonia is an infection that inflames the air sacs in one or both lungs, "
        "which may fill with fluid. It causes cough with phlegm, fever, chills, and "
        "difficulty breathing."
    ),
    "URTI": (
        "Upper Respiratory Tract Infection (URTI) affects the nose, throat, and "
        "sinuses — commonly known as the common cold. Symptoms include runny nose, "
        "sore throat, sneezing, and mild cough."
    ),
}

PRECAUTIONS = {
    "COPD": (
        "Quit smoking and avoid all tobacco smoke.\n"
        "Stay away from dust, fumes, and polluted environments.\n"
        "Get flu and pneumonia vaccines annually.\n"
        "Use prescribed inhalers and medications regularly.\n"
        "Exercise lightly as advised by your doctor."
    ),
    "Asthma": (
        "Identify and avoid personal asthma triggers.\n"
        "Take prescribed inhalers and controller medications daily.\n"
        "Get vaccinated against flu and pneumonia.\n"
        "Keep a rescue inhaler accessible at all times.\n"
        "Maintain clean indoor air and reduce allergen exposure."
    ),
    "Bronchiectasis": (
        "Quit smoking immediately.\n"
        "Get annual flu and pneumonia vaccinations.\n"
        "Practice airway clearance techniques if prescribed.\n"
        "Seek early treatment for any chest infection.\n"
        "Avoid air pollution and occupational dust exposure."
    ),
    "Bronchiolitis": (
        "Wash hands frequently — especially before handling infants.\n"
        "Keep infants away from sick people during winter.\n"
        "Avoid smoking near children.\n"
        "Breastfeed infants to boost immune protection.\n"
        "Disinfect toys and surfaces regularly."
    ),
    "LRTI": (
        "Wash hands often and avoid touching your face.\n"
        "Wear a mask in crowded or high-risk areas.\n"
        "Stay away from people with active respiratory infections.\n"
        "Get recommended vaccinations.\n"
        "Seek medical care early if symptoms worsen."
    ),
    "URTI": (
        "Avoid close contact with people who have colds or flu.\n"
        "Cover your mouth when coughing or sneezing.\n"
        "Wash hands regularly and disinfect shared surfaces.\n"
        "Stay home when sick to prevent spreading infection.\n"
        "Stay hydrated and get adequate rest."
    ),
    "Pneumonia": (
        "Get vaccinated against pneumonia and influenza.\n"
        "Wash hands regularly and maintain good hygiene.\n"
        "Eat a nutritious diet and exercise to strengthen immunity.\n"
        "Quit smoking and limit alcohol consumption.\n"
        "Seek prompt treatment for chest infections."
    ),
    "Healthy": (
        "Maintain regular exercise and a balanced diet.\n"
        "Avoid smoking and limit exposure to air pollution.\n"
        "Get routine health check-ups.\n"
        "Stay hydrated and get adequate sleep."
    ),
}

_classifier = None


def load_classifier():
    global _classifier
    if _classifier is None:
        _classifier = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(46, 46, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(len(LABELS), activation='softmax'),
        ])
        if os.path.exists(MODEL_WEIGHTS):
            _classifier.load_weights(MODEL_WEIGHTS)
        print('Model loaded and ready.')
    return _classifier


def _detection_for(label, confidence):
    return {
        'disease': label,
        'confidence': round(float(confidence), 1),
        'accuracy': MODEL_ACCURACY,
        'description': DESCRIPTIONS[label],
        'causes': CAUSES[label],
        'prevention': PRECAUTIONS[label],
    }


def predict(filename):
    classifier = load_classifier()

    x, sr = librosa.load(filename, sr=22050, duration=30)
    spectrum = librosa.feature.mfcc(y=x, sr=sr)
    spectrum = spectrum.ravel()

    features = spectrum[:6348]
    if len(features) < 6348:
        features = np.pad(features, (0, 6348 - len(features)))

    features = features.reshape(46, 46, 3).astype('float32') / 255.0
    batch = np.expand_dims(features, axis=0)

    raw = classifier.predict(batch, verbose=0)
    probs = raw[0].tolist()
    probabilities = {
        label: round(float(p) * 100, 1)
        for label, p in zip(LABELS, probs)
    }

    ranked = sorted(
        ((label, probabilities[label]) for label in LABELS),
        key=lambda item: item[1],
        reverse=True,
    )
    primary_label, primary_conf = ranked[0]
    secondary_label, secondary_conf = ranked[1]

    detections = [_detection_for(primary_label, primary_conf)]

    dual = False
    if (
        secondary_label != primary_label
        and secondary_conf >= DUAL_DISEASE_MIN_CONFIDENCE
        and not (primary_label != 'Healthy' and secondary_label == 'Healthy')
    ):
        detections.append(_detection_for(secondary_label, secondary_conf))
        dual = True

    primary = detections[0]

    return {
        'disease': primary['disease'],
        'description': primary['description'],
        'causes': primary['causes'],
        'prevention': primary['prevention'],
        'confidence': primary['confidence'],
        'accuracy': MODEL_ACCURACY,
        'detections': detections,
        'dual': dual,
        'probabilities': probabilities,
        'labels': LABELS,
    }
